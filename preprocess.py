import pathlib
import sys
import argparse
import tqdm
from ass import mask2ass
from mask_util import MaskUtil

sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())

def preprocess(
        source_dir: pathlib.Path,
        mask_breath: bool, 
        use_fbl: bool,
        inverse_mask: bool,
):
	audio_list = sorted(source_dir.rglob("audio.wav"))
	
	mask_gen = MaskUtil(
		fbl_onnx_path='./pretrain/fbl_model/model_slim.onnx', 
		mask_breath=mask_breath, 
		use_fbl=use_fbl, 
		inverse_mask=inverse_mask
	)
	
	with tqdm.tqdm(audio_list) as bar:
		for audio_file in bar:
			mask = mask_gen.build_mask(wav_path=audio_file.as_posix())
			mask2ass(mask, audio_file.with_name("mask.ass"))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--source_dir', required=True, help='Path to the directory containing the files')
	parser.add_argument('--mask_breath', action='store_true', help='mask the breath')
	parser.add_argument('--use_fbl', action='store_true', help='use the FoxBreatheLabeler')
	parser.add_argument('--inverse_mask', action='store_true', help='use the inverse mask')
	args = parser.parse_args()
	source_dir_path = pathlib.Path(args.source_dir)
	preprocess(source_dir_path, args.mask_breath, args.use_fbl, args.inverse_mask)
	

if __name__ == "__main__":
	main()
