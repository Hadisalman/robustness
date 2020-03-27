import os
import subprocess
import sys

if __name__ == "__main__":
    bashCommand = "python -m robustness.main " + ' '.join(sys.argv[1:])
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(output)
    print(error)
    # from IPython import embed
    # embed()
    