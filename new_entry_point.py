import subprocess
import sys

if __name__ == '__main__':

    # result = subprocess.run(f'bash -c "conda activate unsupervised_learning; python run.py {" ".join(sys.argv[1:])}"',
    #                         shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    command = f'bash -c "conda activate unsupervised_learning; python run.py {" ".join(sys.argv[1:])}"'
    output = subprocess.Popen(args=command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    while True:
        out = output.stdout.readline().decode("UTF-8")
        if out == '' and output.poll() is not None:
            break
        print(out, flush=True)

