import numpy as np
from scipy.ndimage import center_of_mass
from typing import List
import math

marble_block_1 = np.load(file='C:\Shamini\Assignments\PR\Data\marble_block_1.npy').astype("float64")
marble_block_2 = np.load(file='C:\Shamini\Assignments\PR\Data\marble_block_2.npy').astype("float64")
shape_1 = np.load(file='C:\Shamini\Assignments\PR\Data\shape_1.npy').astype("float64")

def get_orientations_possible(block: np.ndarray) -> List[List[dict]]:
    """Given a 3D numpy array, look at its shape to determine how many ways it
    can be rotated in each axis to end up with a (theoretically) different array
    that still has the SAME shape.

    if all three dimensions are different sizes, then we have 3 more
    orientations, excluding the original, which are all 180-degree rotations.

    if just two dimensions match size, we have 7 plus original. 90-degree
    rotations are around the unique-length axis.

    if all three dimensions match (a cube), then we have 23 plus original.

    :param block: a numpy array of 3 dimensions.
    :return: a list of the ways we can rotate the block. Each is a list of dicts containing parameters for rot90()

    >>> a = np.arange(64, dtype=int).reshape(4, 4, 4)  # a cube
    >>> rotations = get_orientations_possible(a)
    >>> len(rotations)
    23
    >>> rotations  # doctest: +ELLIPSIS
    [[{'k': 1, 'axes': (0, 1)}], ... [{'k': 3, 'axes': (1, 2)}, {'k': 3, 'axes': (0, 2)}]]
    >>> a = a.reshape(2, 4, 8)
    >>> len(get_orientations_possible(a))
    3
    >>> a = a.reshape(16, 2, 2)
    >>> len(get_orientations_possible(a))
    7
    >>> get_orientations_possible(np.array([[1, 2], [3, 4]]))
    Traceback (most recent call last):
    ValueError: array parameter block must have exactly 3 dimensions.
    >>> marble_block_1 = np.load(file='data/marble_block_1.npy')
    >>> len(get_orientations_possible(marble_block_1))
    7
    """

    if len(block.shape) != 3:
        raise ValueError('array parameter block must have exactly 3 dimensions.')

    # Create list of the 23 possible 90-degree rotation combinations -- params to call rot90():

    # consider the 3-tuple shape of axes numbered 0, 1, 2 to represent (height, width, depth)
    (height, width, depth) = block.shape

    if height == width == depth:
        poss = [
            [{'k': 1, 'axes': (0, 1)}],  # 1-axis rotations:
            [{'k': 2, 'axes': (0, 1)}],
            [{'k': 3, 'axes': (0, 1)}],
            [{'k': 1, 'axes': (0, 2)}],
            [{'k': 2, 'axes': (0, 2)}],
            [{'k': 3, 'axes': (0, 2)}],
            [{'k': 1, 'axes': (1, 2)}],
            [{'k': 2, 'axes': (1, 2)}],
            [{'k': 3, 'axes': (1, 2)}],
            [{'k': 1, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],  # 2-axis rotations:
            [{'k': 1, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 2)}],
            [{'k': 1, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}],
            [{'k': 2, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],
            [{'k': 2, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}],
            [{'k': 3, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],
            [{'k': 3, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 2)}],
            [{'k': 3, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}],
            [{'k': 1, 'axes': (1, 2)}, {'k': 1, 'axes': (0, 2)}],
            [{'k': 1, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}],
            [{'k': 1, 'axes': (1, 2)}, {'k': 3, 'axes': (0, 2)}],
            [{'k': 3, 'axes': (1, 2)}, {'k': 1, 'axes': (0, 2)}],
            [{'k': 3, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}],
            [{'k': 3, 'axes': (1, 2)}, {'k': 3, 'axes': (0, 2)}],
        ]
        # return all possibilities, it's a cube

    elif height != width != depth:
        poss = [[{'k': 2, 'axes': (0, 1)}],
                [{'k': 2, 'axes': (0, 2)}],
                [{'k': 2, 'axes': (1, 2)}]
                ]



    elif height != width == depth:
        poss = [
            [{'k': 2, 'axes': (0, 1)}],
            [{'k': 2, 'axes': (0, 2)}],
            [{'k': 1, 'axes': (1, 2)}],
            [{'k': 2, 'axes': (1, 2)}],
            [{'k': 3, 'axes': (1, 2)}],
            [{'k': 1, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}],
            [{'k': 3, 'axes': (1, 2)}, {'k': 2, 'axes': (0, 2)}]
        ]


    elif height == width != depth:
        poss = [
            [{'k': 1, 'axes': (0, 1)}],
            [{'k': 2, 'axes': (0, 1)}],
            [{'k': 3, 'axes': (0, 1)}],
            [{'k': 1, 'axes': (0, 2)}],
            [{'k': 2, 'axes': (1, 2)}],
            [{'k': 1, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 2)}],
            [{'k': 3, 'axes': (0, 1)}, {'k': 2, 'axes': (0, 2)}]]



    elif height == depth != width:

        poss = [[{'k': 2, 'axes': (0, 1)}],
                [{'k': 1, 'axes': (0, 2)}],
                [{'k': 2, 'axes': (0, 2)}],
                [{'k': 2, 'axes': (0, 2)}],
                [{'k': 2, 'axes': (1, 2)}],
                [{'k': 2, 'axes': (0, 1)}, {'k': 1, 'axes': (0, 2)}],
                [{'k': 2, 'axes': (0, 1)}, {'k': 3, 'axes': (0, 2)}]]

    return poss

# returning the possible combinations for given height,width,depth



'''lenb=len(get_orientations_possible(marble_block_1))
print(lenb)
'''

def block_rotation(block:np.array)-> List:
    '''
     trying to find the actual rotation values from the possible orientations we got from get_orientations_possible(block)
     function.
    :param block:  a numpy array of 3 dimensions.
    :return: a list of the ways we can rotate the block using np.rot(90) for each possible orientations.

    '''

    poss = get_orientations_possible(block)

    result=[]
    for i in poss:

        if len(i) == 2:
            rotation_1 = np.rot90(block, k=i[0]['k'], axes=i[0]['axes'])
            rotation_2 = np.rot90(rotation_1, k=i[1]['k'], axes=i[1]['axes'])
            result.append(rotation_2)
        else:
            rotation_3 = np.rot90(block, k=i[0]['k'], axes=i[0]['axes'])
            result.append(rotation_3)

    return (result)


''''
fit=block_rotation(marble_block_1)
print(len(fit))
'''
def carve_sculpture_from_density_block(shape: np.ndarray, block: np.ndarray) -> np.ndarray:

    return np.multiply(shape, block)

    """The shape array guides our carving. It indicates which parts of the
    material block to keep (the 1 values) and which to carve away (the 0 values),
    producing a new array that defines a sculpture and its varying densities.
    It must have NaN values everywhere that was 'carved' away.

    :param shape: array to guide carving into some 3D shape
    :param block: array describing densities throughout the raw material block
    :return: array of densities in the resulting sculpture, in same orientation.
    :raises: ValueError if the input arrays don't match in size and shape.
    """
    # TODO: write the code for this function, which could be as short as one line of code!
    # TODO: Add a few good, working Doctests
def finding_max_density_block(final_result_List: List[np.array]) :
    '''
    In order to find the max average density the 0 in the block file is replaced with nan

    :param final_result_List: List of numpy array which is result of carving block according to the shape.
    :return: maximum of list of values
    '''
    max_density=[]
    for i in final_result_List:
        replacing_nan = np.where(i == 0, np.nan, i)
        max_density.append(np.nanmean(replacing_nan))
        #if i == 0:
            #max_density.append(nan)
        return (max(max_density))

def finding_centre_of_mass(final_result_List: List[np.array])-> List:
    '''
    Finding the centre of mass for the carved sculpture to find the stable state.
    :param final_result_List:List of numpy array which is result of carving block according to the shape.
    :return: List of centre of mass value calculated from the final_result_list
    '''

    centre_of_mass_list=[]
    for i in final_result_List:
       centre_of_mass_list.append(center_of_mass(i))

    return centre_of_mass_list





block_result1=[]
for i in block_rotation(marble_block_1):
    block_result1.append(i)

block_result2=[]
for i in block_rotation(marble_block_2):
    block_result2.append(i)

final_result1=[]
for i in block_result1:
    density=carve_sculpture_from_density_block(shape_1,i)
    final_result1.append(density)


final_result2=[]
for i in block_result2:
    density=carve_sculpture_from_density_block(shape_1,i)
    final_result2.append(density)

print("The mean density of ")
if finding_max_density_block(final_result1) > finding_max_density_block(final_result2):
    print(finding_max_density_block(final_result1))

else:
    print(finding_max_density_block(final_result2))

while True:
    print("THe centre of mass calculation")
    print(finding_centre_of_mass(final_result1))
    print(finding_centre_of_mass(final_result2))


'''
Need to improve code in finding the stability function using the centre mass calcultion.
Since I had a "data cannot be loaded error" I couldn't see my results for some functions without which i couldn't move forward.
Also when I added my code to the forked iSchool repository, after clicking "Collaborators & teams: I couldn't see "student team" 
so i couldn't remove it. 
 The code can also be viewed in my Github: shaminiGit
'''

