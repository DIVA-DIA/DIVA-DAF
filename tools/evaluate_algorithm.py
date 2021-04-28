import argparse
import itertools
import re
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT

import numpy as np

from tools.utils.overall_score import write_stats

EXTENSIONS_PATTERNS = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg']


def check_extension(filename, extension_list):
    return any(filename.endswith(extension) for extension in extension_list)


def get_file_list(path_to_folder: Path, extension_list):
    file_paths = [f for p in extension_list for f in path_to_folder.glob(p)]
    file_paths.sort()
    return file_paths


def get_score(logs, token):
    for line in logs:
        line = str(line)
        if token in line:
            split = line.split('=')[1][0:8]
            split = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", split)
            return float(split[0])
    return None


def evaluate(prediction_img_path: Path, gt_img_path: Path, output_path: Path, eval_tool_path: Path,
             original_img_path: Path, no_visualization: bool):
    print("Starting: JAR {}".format(prediction_img_path.name))
    command = ['java', '-jar', str(eval_tool_path),
               '-gt', str(gt_img_path),
               '-p', str(prediction_img_path),
               '-out', str(output_path)]
    if original_img_path is not None:
        command.append(['-o', str(original_img_path)])
    if no_visualization:
        command.append('-dv')
    else:
        # Check if output path exists
        (prediction_img_path.parent / output_path).mkdir(parents=True, exist_ok=True)

    p = Popen(command, stdout=PIPE, stderr=STDOUT)
    logs = [line for line in p.stdout]
    print("Done: JAR {}".format(prediction_img_path.name))
    return [get_score(logs, 'Mean IU (Jaccard index) = '), logs]


def main(gt_folder: Path, prediction_folder: Path, original_images: str, output_path: Path, no_visualization: bool,
         eval_tool: Path, processes: int):
    # Select the number of threads
    if processes == 0:
        pool = Pool(processes=cpu_count())
    else:
        pool = Pool(processes=processes)

    # Get the paths for all gt files
    gt_files_path = get_file_list(gt_folder, EXTENSIONS_PATTERNS)

    # Get the the paths for all prediction images
    prediction_files_path = get_file_list(prediction_folder, EXTENSIONS_PATTERNS)

    # Check if we have the same amount of gt and prediction files
    assert len(gt_files_path) == len(prediction_files_path), "Amount of gt files and prediction files differ."

    # Get the paths of original images if set
    if original_images is not None:
        original_images = Path(original_images)
        original_files_path = get_file_list(Path(original_images), EXTENSIONS_PATTERNS)
    else:
        original_files_path = [None] * len(gt_files_path)

    # Timer
    tic = time.time()

    # Debugging purposes only!
    # input_images = [input_images[1]]
    # gt_xml = [gt_xml[1]]
    # gt_pxl = [gt_pxl[1]]

    # input_xml = input_xml[0:3]
    # gt_xml = gt_xml[0:3]
    # gt_pxl = gt_pxl[0:3]

    # For each file run
    results = list(pool.starmap(evaluate, zip(prediction_files_path,
                                              gt_files_path,
                                              itertools.repeat(output_path),
                                              itertools.repeat(eval_tool),
                                              original_files_path,
                                              itertools.repeat(no_visualization)
                                              )))
    pool.close()
    print("Pool closed)")

    scores = []
    errors = []

    for item in results:
        if item[0] is not None:
            scores.append(item[0])
        else:
            errors.append(item)

    if list(scores):
        score = np.mean(scores)
    else:
        score = -1

    # np.save(os.path.join(output_path, 'results.npy'), results)
    write_stats(results, errors, score)
    print('Total time taken: {:.2f}, avg_miou={}, nb_errors={}'.format(time.time() - tic, score, len(errors)))
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grid search to identify best hyper-parameters for text line '
                                                 'segmentation.')
    # Path folders (evaluator)
    parser.add_argument('--gt_folder', type=Path,
                        required=True,
                        help='path to folders containing the gt images (e.g. /dataset/CB55/test-page).'
                             ' Will fail if path contains spaces!')
    parser.add_argument('--prediction_folder', type=Path,
                        required=True,
                        help='path to folders containing prediction images (e.g. /dataset/CB55/test-m).'
                             ' Will fail if path contains spaces!')
    parser.add_argument('--original_images', type=str, default=None,
                        help="Path to the folder containing the original rgb images. If set will create overlap images")
    parser.add_argument('--output_path', type=Path,
                        required=True,
                        help='path to store output files RELATIVE TO PREDICTION PATH')
    parser.add_argument('--no_visualization', action='store_true',
                        help="If set the program will not produce a visualization")

    # Method parameters
    # Environment
    parser.add_argument('--eval-tool', type=Path,
                        default='./utils/LayoutAnalysisEvaluator.jar',
                        help='path to folder containing DIVA_Line_Segmentation_Evaluator')
    parser.add_argument('--processes', '-p', type=int,
                        default=0,
                        help='number of thread to use for parallel search. If set to 0 #cores will be used instead')
    args = parser.parse_args()

    main(**args.__dict__)
