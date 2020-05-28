#!/home/draguet/QGdis/QGdis-env/bin/python3
#############################################################################
#
# Submitter_Condor.py
#
# Program to submit the code to Condor.
#
# Author -- Maxence Draguet (26/05/2020) heavily influenced by Aaron O'Neill.
#
#############################################################################
import os

def executor():
    # Set up the path to the environment
    venv_path = '/home/draguet/QGdis/QGdis-env/bin/python3'
    # Set up the path to the code
    code_path = '/home/draguet/QGdis/Code/Experiments/Main.py'
    # Run the Main.py, magically loads everything in the process.
    os.system('{0} {1}'.format(venv_path, code_path))
                    
    print('Completed successfully.')

if __name__ == "__main__":
    executor()
