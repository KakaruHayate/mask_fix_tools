import os
import pathlib

import numpy as np
import onnxruntime as ort
import torchaudio
import yaml

from funasr import AutoModel
from icecream import ic

def load_config_from_yaml(file_path):
	with open(file_path, 'r') as file:
		config = yaml.safe_load(file)
	return config


class MaskUtil:
	def __init__(self, fbl_onnx_path, ap_threshold=0.4, ap_dur=0.08, use_fbl=True, mask_breath=False, inverse_mask=False):
		super().__init__()
		self.inverse_mask = inverse_mask
		# FSMN-VAD
		self.vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True, log_level="ERROR", disable_pbar=True, disable_log=True)
		# FBL
		self.use_fbl = use_fbl
		self.mask_breath = mask_breath
		self.ap_threshold = ap_threshold
		self.ap_dur = ap_dur
		self.fbl_onnx_path = fbl_onnx_path
		self.fbl_session = ort.InferenceSession(fbl_onnx_path)
		self.fbl_input_name = self.fbl_session.get_inputs()[0].name
		self.fbl_output_name = self.fbl_session.get_outputs()[0].name
		config_file = pathlib.Path(self.fbl_onnx_path).with_name('config.yaml')
		assert os.path.exists(self.fbl_onnx_path), f"Onnx file does not exist: {self.fbl_onnx_path}"
		assert config_file.exists(), f"Config file does not exist: {config_file}"
		self.fbl_config = load_config_from_yaml(config_file)
		self.time_scale = 1.0 / (self.fbl_config['audio_sample_rate'] / self.fbl_config['hop_size'])
		self.mel_transform = torchaudio.transforms.MelSpectrogram(
				sample_rate=self.fbl_config['audio_sample_rate'],
				n_fft=1024,
				hop_length=256,
				n_mels=128
			)

	def find_segments_dynamic(self, arr, threshold=0.5, max_gap=5, ap_threshold=10):
		segments = []
		start = None
		gap_count = 0

		for i in range(len(arr)):
			if arr[i] >= threshold:
				if start is None:
					start = i
				gap_count = 0
			else:
				if start is not None:
					if gap_count < max_gap:
						gap_count += 1
					else:
						end = i - gap_count - 1
						if end >= start and (end - start) >= ap_threshold:
							segments.append((start * self.time_scale, end * self.time_scale))
						start = None
						gap_count = 0

		# Handle the case where the array ends with a segment
		if start is not None and (len(arr) - start) >= ap_threshold:
			segments.append((start * self.time_scale, (len(arr) - 1) * self.time_scale))

		return segments

	def fbl_infer(self, wav_path):
		# 生成AP的MASK段落，因为涉及读取顺便把音频总长计算出来
		audio, sr = torchaudio.load(wav_path)
		wave_length = audio.size(1) / sr
		audio = audio[0][None, :]
		if sr != self.fbl_config['audio_sample_rate']:
			audio = torchaudio.transforms.Resample(sr, self.fbl_config['audio_sample_rate'])(audio)
		
		mel_spectrogram = self.mel_transform(audio).squeeze().numpy()
		ap_probability = self.fbl_session.run([self.fbl_output_name], {self.fbl_input_name: [audio[0].numpy()]})[0]
		sxp = ap_probability[0]
		
		breath_segments = self.find_segments_dynamic(sxp, threshold=self.ap_threshold,
										 ap_threshold=int(self.ap_dur / self.time_scale))

		return breath_segments, wave_length
	
	def vad_infer(self, wav_path):
		# VAD的推理
		sil_segments = self.vad_model.generate(input=wav_path)
		
		return sil_segments
	
	def calculate_silence_segments(self, speech_segments, total_duration):
		# 处理VAD输出获得静音MASK段落，FSMN-VAD的输出单位是ms
		silence_segments = []
		previous_end = 0

		time_stamps = speech_segments[0]['value'] if speech_segments else []

		for start_ms, end_ms in time_stamps:
			start = start_ms / 1000.0
			end = end_ms / 1000.0
			if previous_end < start:
				silence_segments.append((previous_end, start))
			previous_end = end

		if previous_end < total_duration:
			silence_segments.append((previous_end, total_duration))

		return silence_segments

	def merge_silence_segments(self, merged_segments):
		# 输入前把所有的mask段落都加起来
		# 合并交叠的mask片段
		merged_segments.sort(key=lambda x: x[0])
		merged_silence_segments = []
		for start, end in merged_segments:
			if not merged_silence_segments or merged_silence_segments[-1][1] < start:
				merged_silence_segments.append((start, end))
			else:
				merged_silence_segments[-1] = (merged_silence_segments[-1][0], max(merged_silence_segments[-1][1], end))

		return merged_silence_segments

	def remove_segments_from_a_by_b(self, a_segments, b_segments):
		# 从segments a中删除segments b包含的部分
		merged_b_segments = self.merge_silence_segments(b_segments)
		result_segments = []
		for a_start, a_end in a_segments:
			current_start = a_start
			for b_start, b_end in merged_b_segments:
				if a_end <= b_start:
					result_segments.append((current_start, a_end))
					break
				elif a_start >= b_end:
					continue
				else:
					if current_start < b_start:
						result_segments.append((current_start, b_start))
					current_start = b_end
					if current_start >= a_end:
						break
			if current_start < a_end:
				result_segments.append((current_start, a_end))
		
		return result_segments

	def build_mask(self, wav_path):
		# FBL，检测数据中换气声的片段，需要以此进行后面的处理
		if self.use_fbl:
			breath_segments, wave_length = self.fbl_infer(wav_path)
		# VAD，数据的静音部分仍有读数，这些是不需要的
		silence_segments = self.vad_infer(wav_path)
		silence_segments = self.calculate_silence_segments(silence_segments, wave_length)
		# 可能需要预测换气部分的Ope，也可能不需要，但总之需要处理一下一半一半的问题
		if self.use_fbl:
			if self.mask_breath:
				mask_segments += breath_segments
			else:
				mask_segments = self.remove_segments_from_a_by_b(silence_segments, breath_segments)
			# 合并
			mask_segments = self.merge_silence_segments(mask_segments)
		else:
			mask_segments = silence_segments

		if self.inverse_mask:
			mask_segments = 1 - mask_segments
		
		return mask_segments

# import argparse
# from icecream import ic
# def main():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--audio_path', required=True, help='Path to the directory containing the audio files')
# 	args = parser.parse_args()
# 	
# 	mask_gen = MaskUtil(fbl_onnx_path='./pretrain/fbl_model/model_slim.onnx')
# 	
# 	mask = mask_gen.build_mask(args.audio_path)
# 	
# 	ic(mask)
# 	
# if __name__ == '__main__':
# 	main()
