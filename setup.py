from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'

# function to read the requirements from file
def get_requirements(file_path:str)->List[str]:
    '''
    This function return the list of requirements
    
    :param file_path: Description
    :type file_path: str
    :return: Description
    :rtype: List[str]
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements



setup(
	name='mlproject',
	version='0.0.1',
	author='Ankit Sharma',
	author_email='ankit302290@gmail.com',
	packages=find_packages(),
	install_requires=get_requirements('requirements.txt')
)