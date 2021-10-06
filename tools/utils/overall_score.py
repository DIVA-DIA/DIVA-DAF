from pathlib import Path
from typing import List, Any


def write_stats(results: List[Any], errors: List[Any], score):
    if not results:
        return

    headers, stat_list, output_path = create_stat_list(results)
    stat_list.append(['Total IU', '', '', str(score)])

    with (output_path / 'summary.csv').open(mode='w') as f:
        # write headers
        f.write('filename,' + ','.join(headers) + '\n')
        for i, line in enumerate(stat_list):
            f.write((','.join(line)) + '\n')

    if not errors:
        return

    with (output_path / 'error_log.txt').open(mode='w') as f:
        for error in errors:
            f.writelines([line.decode('ascii') for line in error[1]])
            f.write("\n--------------------------------------------------\n\n")


def create_stat_list(result_list: List[Any]):
    result_list = clean_result_list(result_list)
    headers = [header.split('=')[0] for header in result_list[0][0].split(' ')]
    # interate over the list
    for i, result in enumerate(result_list):
        # get file name
        filename = result[1].stem[:-14]
        # get the metrices (EM, HS, IU, F1, P, R, Freq) [freq has no space before]
        metric_list_strings = result[0].split(' ')
        metric_list_numbers = [m.split('=')[-1] for m in metric_list_strings]
        result_list[i] = [filename] + metric_list_numbers
    return headers, result_list, result[1].parent


def clean_result_list(results: List[Any]):
    for i, result in enumerate(results):
        # convert to string
        page_stats_string = result[1][1].decode('UTF-8')
        page_path = Path(result[1][2].decode('UTF-8').split(' ')[4])
        # change string at index "Freq" by adding a space
        index_start_freq = page_stats_string.find('Freq')
        page_stats_string = page_stats_string[:index_start_freq] + ' ' + page_stats_string[index_start_freq:]
        page_stats_string = page_stats_string.replace(':', '=').replace('\n', '').replace(',', ';')
        results[i] = [page_stats_string, page_path]
    return results


if __name__ == '__main__':
    write_stats(path=Path('.'), results=[[0.97789, [b'Mean IU (Jaccard index) = 0.97789\n',
                                                    b'EM=1.00 HS=1.00 IU=0.98,1.00[1.00|0.99|0.94|0.99] F1=0.99,1.00[1.00|0.99|0.97|0.99] P=1.00,1.00[1.00|1.00|0.99|1.00] R=0.98,1.00[1.00|0.99|0.94|0.99]Freq:[0.84|0.06|0.01|0.09]\n',
                                                    b'Visualization image written in: /Users/voegtlil/Desktop/test/output/../eval_output_test/output_e-codices_fmb-cb-0055_0098v_max.visualization.png\n']],
                                         [0.97789, [b'Mean IU (Jaccard index) = 0.97789\n',
                                                    b'EM=1.00 HS=1.00 IU=0.98,1.00[1.00|0.99|0.94|0.99] F1=0.99,1.00[1.00|0.99|0.97|0.99] P=1.00,1.00[1.00|1.00|0.99|1.00] R=0.98,1.00[1.00|0.99|0.94|0.99]Freq:[0.84|0.06|0.01|0.09]\n',
                                                    b'Visualization image written in: /Users/voegtlil/Desktop/test/output/../eval_output_test/output_e-codices_fmb-cb-0055_0098v_max.visualization.png\n']]],
                errors=[], score=-1)
