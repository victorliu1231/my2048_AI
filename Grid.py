from cgi import test
from turtle import backward
from unittest import skip
import numpy as np
import random

def score(grid):
    '''Returns the total score in a grid.'''
    return np.sum(grid)

def is_equal_arrays(array1, array2):
    '''Determines whether array1 and array2 are equal element-wise.'''
    return (array1 == array2).all()

def is_all_arrays_equal(list_of_arrs):
    '''Determines whether all arrays in the list of arrays are equal element-wise.'''
    if len(list_of_arrs) == 1:
        return True
    return True if len(np.unique(list_of_arrs, axis=0)) == 1 else False

def remove_inner_zeros(array_1d):
    '''Removes any inner zeros in a 1D array.'''
    no_inner_zeroes_array_1d = array_1d[array_1d != 0]
    pad_length = len(array_1d) - len(no_inner_zeroes_array_1d)
    no_inner_zeroes_array_1d = np.pad(no_inner_zeroes_array_1d, (0, pad_length), "constant")
    return no_inner_zeroes_array_1d

def shift_cells_left_helper(grid):
    '''Helper function to shift cells left using 2048-like rules.'''
    new_grid = np.copy(grid)
    grid_shape = new_grid.shape
    for i in range(grid_shape[0]): #indexes by rows
        for j in range(grid_shape[1]): #indexes by entries in each row
            if j != grid_shape[1]-1: #ensures that j+1 does not go out of bounds
                new_grid[i][:] = remove_inner_zeros(new_grid[i][:]) #shift over cells to the left to remove any inner zeros in the beginning
                if (new_grid[i][j] == new_grid[i][j+1]): #if this tile is the same number as the tile to its right and it's not zero
                    new_grid[i][j] = new_grid[i][j] + new_grid[i][j+1] #then add the two tiles together and place the sum one tile to the left
                    new_grid[i][j+1] = 0 #and set the right tile to zero

                #remove any inner zeros (will shift all tiles to the left if there is a free cell in this row)
                new_grid[i][:] = remove_inner_zeros(new_grid[i][:])
    return new_grid

def num_adj_tiles_of_number(grid, number):
    '''Returns the total number of pairs of mergeable adjacent tiles of the specified number.'''
    # Counts the total number of pairs of mergeable adjacent side-to-side tiles
    num_side_to_side_adj_tiles = 0
    grid_shape = grid.shape
    skip_next_side_to_side_tile = False #initialization
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # If find an adjacent side-to-side tile, add 1 to the adjacent side-to-side tile counter
            #  and skip the next side-to-side tile. Else, don't skip the next side-to-side tile.
            if j != grid_shape[1] - 1:
                if (grid[i][j] == grid[i][j+1]) & (grid[i][j] == number) & (not skip_next_side_to_side_tile):
                    num_side_to_side_adj_tiles = num_side_to_side_adj_tiles + 1
                    skip_next_side_to_side_tile = True
                else:
                    skip_next_side_to_side_tile = False
            else:
                skip_next_side_to_side_tile = False
    
    # Counts the total number of pairs of mergeable up-and-down adjacent tiles
    num_up_and_down_adj_tiles = 0
    transpose_grid = np.transpose(grid)
    transpose_grid_shape = transpose_grid.shape
    skip_next_up_and_down_tile = False #initialization
    for i in range(transpose_grid_shape[0]):
        for j in range(transpose_grid_shape[1]):
            # If find an adjacent up-and-down tile, add 1 to the adjacent up-and-down tile counter
            #  and skip the next up-and-down tile. Else, don't skip the next up-and-down tile.
            if j != transpose_grid_shape[1] - 1:
                if (transpose_grid[i][j] == transpose_grid[i][j+1]) & (transpose_grid[i][j] == number) & (not skip_next_up_and_down_tile):
                    num_up_and_down_adj_tiles = num_up_and_down_adj_tiles + 1
                    skip_next_up_and_down_tile = True
                else:
                    skip_next_up_and_down_tile = False
            else:
                skip_next_up_and_down_tile = False

    # Determines whether the number of pairs of mergeable adjacent tiles in greater in the side-to-side
    #   direction or the up-and-down direction. Returns the total number of pairs of mergeable adjacent 
    #   tiles in the direction with the most pairs of mergeable adjacent tiles.
    array_num_adj_tiles_both_dir = np.array([num_side_to_side_adj_tiles, num_up_and_down_adj_tiles])
    max_num_adj_tiles = max(num_side_to_side_adj_tiles, num_up_and_down_adj_tiles)
    return max_num_adj_tiles

def num_adj_tiles(grid):
    '''Returns the total number of pairs of mergeable adjacent tiles.'''
    # Counts the total number of pairs of mergeable adjacent side-to-side tiles
    num_side_to_side_adj_tiles = 0
    grid_shape = grid.shape
    skip_next_side_to_side_tile = False #initialization
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            # If find an adjacent side-to-side tile, add 1 to the adjacent side-to-side tile counter
            #  and skip the next side-to-side tile. Else, don't skip the next side-to-side tile.
            if j != grid_shape[1] - 1:
                if (grid[i][j] == grid[i][j+1]) & (not skip_next_side_to_side_tile) & (grid[i][j] != 0):
                    num_side_to_side_adj_tiles = num_side_to_side_adj_tiles + 1
                    skip_next_side_to_side_tile = True
                else:
                    skip_next_side_to_side_tile = False
            else:
                skip_next_side_to_side_tile = False
    
    # Counts the total number of pairs of mergeable up-and-down adjacent tiles
    num_up_and_down_adj_tiles = 0
    transpose_grid = np.transpose(grid)
    transpose_grid_shape = transpose_grid.shape
    skip_next_up_and_down_tile = False #initialization
    for i in range(transpose_grid_shape[0]):
        for j in range(transpose_grid_shape[1]):
            # If find an adjacent up-and-down tile, add 1 to the adjacent up-and-down tile counter
            #  and skip the next up-and-down tile. Else, don't skip the next up-and-down tile.
            if j != transpose_grid_shape[1] - 1:
                if (transpose_grid[i][j] == transpose_grid[i][j+1]) & (not skip_next_up_and_down_tile) & (transpose_grid[i][j] != 0):
                    num_up_and_down_adj_tiles = num_up_and_down_adj_tiles + 1
                    skip_next_up_and_down_tile = True
                else:
                    skip_next_up_and_down_tile = False
            else:
                skip_next_up_and_down_tile = False

    # Determines whether the number of pairs of mergeable adjacent tiles in greater in the side-to-side
    #   direction or the up-and-down direction. Returns the total number of pairs of mergeable adjacent 
    #   tiles in the direction with the most pairs of mergeable adjacent tiles.
    array_num_adj_tiles_both_dir = np.array([num_side_to_side_adj_tiles, num_up_and_down_adj_tiles])
    max_num_adj_tiles = max(num_side_to_side_adj_tiles, num_up_and_down_adj_tiles)
    return max_num_adj_tiles

def free_cells(grid):
    """Returns a list of empty cells."""
    grid_shape = grid.shape
    return [(x, y)
        for x in range(grid_shape[1])
        for y in range(grid_shape[0])
        if not grid[y][x]]

def num_free_cells(grid):
    '''Returns the total number of free cells.'''
    return len(free_cells(grid))    

def num_inst_in_grid(grid, inst):
    '''Returns the total number of instances 'inst' in a 2D array.'''
    return np.isclose(grid, inst).sum()

def kth_max(grid, k):
    '''Returns the kth largest value in a 2D array.'''
    return np.unique(grid)[-1*k]

def is_max_tile_in_corner(grid, return_type):
    '''Returns whether a max tile is in the corner of the 2D array or not.'''
    max_tile = np.amax(grid)
    if return_type == "boolean":
        return(grid[0][0] == max_tile or grid[0][-1] == max_tile or grid[-1][0] == max_tile or grid[-1][-1] == max_tile)
    elif return_type == "list":
        max_tile_corner_coords = []
        if grid[0][0] == max_tile:
            max_tile_corner_coords.append([0, 0])
        if grid[0][-1] == max_tile:
            max_tile_corner_coords.append([0, -1])
        if grid[-1][0] == max_tile:
            max_tile_corner_coords.append([-1, 0])
        if grid[-1][-1] == max_tile:
            max_tile_corner_coords.append([-1, -1])
        return max_tile_corner_coords
    else:
        raise Exception("Input Error: 'return_type' for this function should either be 'boolean' or 'list'.")

def tile_coords_of_number_along_edge(grid, number):
    '''Returns the coordinates of all instances of a number along the edge in a 2D array.'''
    numpy_list_num_tile_coords = np.where(grid == number)
    python_list_num_tile_coords = []
    num_rows, num_cols = grid.shape[0], grid.shape[1]
    for i in range(len(numpy_list_num_tile_coords[0])):
        row, col = numpy_list_num_tile_coords[0][i], numpy_list_num_tile_coords[1][i]
        if (row == 0 or row == num_rows-1) or (col == 0 or col == num_cols-1):
            python_list_num_tile_coords.append([row, col])
    return python_list_num_tile_coords

def is_two_coords_adj(y1, x1, y2, x2, direction):
    '''Determines if two coordinates are adjacent to one another on a 2D grid in the specified direction.'''
    if direction == "horizontal":
        return abs(x1 - x2) == 1
    elif direction == "vertical":
        return abs(y1- y2) == 1
    else:
        raise Exception("Input Error: this function can only take in 'horizontal' or 'vertical' for direction.")

def num_number_adj_to_tile(grid, number, tile_coords):
    '''Determines the number of instances where a specified number is adjacent to a specified tile on a 2D grid.
        For the purposes of the AI, this function can only accept tile_coords along the edge of the board, and will
        only consider numbers along the edge of the board as truly adjacent. It will search along the horizontal
        and the vertical direction and output the total number of adjacent tiles in both directions.'''
    num_rows, num_cols = grid.shape[0], grid.shape[1]
    if num_rows * num_cols < 2:
        raise Exception("Input Error: The input array to this function should be a 2D grid of at least 2 tiles.")
    row_tile, col_tile = tile_coords[0], tile_coords[1]
    if (row_tile == 0 or row_tile == num_rows-1) or (col_tile == 0 or col_tile == num_cols-1):
        numpy_list_num_coords = np.where(grid == number)
        if len(numpy_list_num_coords[0]) == 0:
            return 0
        horizontal_counter = 0
        vertical_counter = 0
        for i in range(len(numpy_list_num_coords[0])):
            row_num, col_num = numpy_list_num_coords[0][i], numpy_list_num_coords[1][i]
            if (row_num == 0 or row_num == num_rows-1) or (col_num == 0 or col_num == num_cols-1):
                if is_two_coords_adj(row_num, col_num, row_tile, col_tile, "horizontal"):
                    horizontal_counter = horizontal_counter + 1
                if is_two_coords_adj(row_num, col_num, row_tile, col_tile, "vertical"):
                    vertical_counter = vertical_counter + 1
        return horizontal_counter + vertical_counter
    else:
        raise Exception("Input Error: This function can only accept tile_coords that are along the edges of the grid.")

def num_edges_from_a_corner_max_tile_in_sorted_order(grid, corner_coord):
    '''Determines the number of sorted edges emanating from a corner max tile. If the inputted coordinates
        are not in a corner or the inputted coords are not pointing to a max tile, return 0.
        Can take in both one corner coord or a list of corner coords.'''
    num_rows, num_cols = grid.shape[0], grid.shape[1]
    max_num = np.amax(grid)
    transposed_grid = np.transpose(grid)
    sorted_edges = 0
    if np.array(corner_coord).ndim == 1:
        if len(corner_coord) == 0:
            return 0
        row_corner_tile, col_corner_tile = corner_coord[0], corner_coord[1]
        if (row_corner_tile == 0 or row_corner_tile == num_rows-1) and (col_corner_tile == 0 or col_corner_tile == num_cols-1) and (grid[row_corner_tile, col_corner_tile] == max_num):
            forward_horizontal_sort = np.sort(grid[row_corner_tile])
            backward_horizontal_sort = -np.sort(-grid[row_corner_tile])
            if np.array_equal(grid[row_corner_tile], forward_horizontal_sort) or np.array_equal(grid[row_corner_tile], backward_horizontal_sort):
                sorted_edges = sorted_edges + 1
            
            forward_vertical_sort = np.sort(transposed_grid[col_corner_tile])
            backward_vertical_sort = -np.sort(-transposed_grid[col_corner_tile])
            if np.array_equal(transposed_grid[col_corner_tile], forward_vertical_sort) or np.array_equal(transposed_grid[col_corner_tile], backward_horizontal_sort):
                sorted_edges = sorted_edges + 1
            return sorted_edges
        else:
            return 0
    elif np.array(corner_coord).ndim == 2:
        for one_corner_coord in corner_coord:
            if len(one_corner_coord) == 0:
                return 0
            row_corner_tile, col_corner_tile = one_corner_coord[0], one_corner_coord[1]
            if (row_corner_tile == 0 or row_corner_tile == num_rows-1) and (col_corner_tile == 0 or col_corner_tile == num_cols-1) and (grid[row_corner_tile, col_corner_tile] == max_num):
                forward_horizontal_sort = np.sort(grid[row_corner_tile])
                backward_horizontal_sort = -np.sort(-grid[row_corner_tile])
                if np.array_equal(grid[row_corner_tile], forward_horizontal_sort) or np.array_equal(grid[row_corner_tile], backward_horizontal_sort):
                    sorted_edges = sorted_edges + 1
                
                forward_vertical_sort = np.sort(transposed_grid[col_corner_tile])
                backward_vertical_sort = -np.sort(-transposed_grid[col_corner_tile])
                if np.array_equal(transposed_grid[col_corner_tile], forward_vertical_sort) or np.array_equal(transposed_grid[col_corner_tile], backward_horizontal_sort):
                    sorted_edges = sorted_edges + 1
        return sorted_edges    

def apply_one_param_func_to_list_of_2D_arrays(list_of_2D_arrays, func):
    list_func_output = []
    for grid in list_of_2D_arrays:
        list_func_output.append(func(grid))
    return np.array(list_func_output)

def apply_two_param_func_to_list_of_2D_arrays(list_of_2D_arrays, second_param, func):
    list_func_output = []
    for grid in list_of_2D_arrays:
        list_func_output.append(func(grid, second_param))
    return np.array(list_func_output)

def apply_three_param_func_to_list_of_2D_arrays(list_of_2D_arrays, second_param, third_param, func):
    list_func_output = []
    for grid in list_of_2D_arrays:
        list_func_output.append(func(grid, second_param, third_param))
    return np.array(list_func_output)

def assign_score(grid):
    MAX_TILE_IN_CORNER_WEIGHT = 1
    MAX_TILE_WEIGHT = 0.075
    TILE_WEIGHTS = {
        0: 0,
        2: 0,
        4: 1,
        8: 3,
        16: 9,
        32: 27,
        64: 81,
        128: 243,
        256: 729,
        512: 2187,
        1024: 6561,
        2048: 19683,
        4096: 59049,
        8192: 177147,
        16384: 531441,
        32768: 1594323,
        65536: 4782969,
        131072: 14348907
    }
    NUM_ADJACENTS_WEIGHT = 0.05
    NUM_SORTED_EDGES_WEIGHT = 0.025
    NUM_FREE_CELLS_WEIGHT = 0.01
    score = 0

    # Multiply the number of max tiles in the corners of the grid by MAX_TILE_IN_CORNER_WEIGHT, 
    #   then add the product to the score.
    corner_max_tile_coords = is_max_tile_in_corner(grid, "list")
    score = score + len(corner_max_tile_coords) * MAX_TILE_IN_CORNER_WEIGHT

    # Multiply the number of tiles of a certain number by its respective tile weight scaled by MAX_TILE_WEIGHT (exception: 4 and 2 tiles), then add the product to the score.
    uniques = np.unique(grid)
    for unique in uniques:
        if unique != 0 and unique != 2 and unique != 4:
            score = score + num_inst_in_grid(grid, unique)*TILE_WEIGHTS[unique]*MAX_TILE_WEIGHT
    
    # Multiply the number of adjacent tiles of each number in the grid by its respective tile weight
    #   scaled by NUM_ADJACENTS_WEIGHT, then add the product to the score.
    uniques = np.unique(grid)
    for unique in uniques:
        score = score + num_adj_tiles_of_number(grid, unique)*TILE_WEIGHTS[unique]*NUM_ADJACENTS_WEIGHT

    # Multiply the number of sorted edges by NUM_SORTED_EDGES_WEIGHT, then add the product to the score.
    max_num = np.amax(grid)
    coords_of_max_num = tile_coords_of_number_along_edge(grid, max_num)
    num_sorted_edges = 0
    for i in range(len(coords_of_max_num)):
        num_sorted_edges = num_sorted_edges + num_edges_from_a_corner_max_tile_in_sorted_order(grid, coords_of_max_num[i])
    score = score + num_sorted_edges*NUM_SORTED_EDGES_WEIGHT

    # Multiply the number of free cells by NUM_FREE_CELLS_WEIGHT, then add the product to the score.
    score = score + num_free_cells(grid)*NUM_FREE_CELLS_WEIGHT

    return score

class Grid:
    def __init__(self, grid):
        self.grid = grid
        self.old_grid = grid

    def shift_cells_left(self, change_grid_state = True):
        '''Shifts cells left using the shift_cells_left_helper function. 
           change_grid_state: states whether the grid state of the Grid object will be changed. 
                If True, returns the new grid state.
                If False, does not change the grid state but returns the hypothetical grid state that
                    would've been generated if change_grid_state was True.
                change_grid_state is True by default'''
        new_grid = shift_cells_left_helper(self.grid)
        
        if change_grid_state == True:
            self.old_grid = self.grid
            self.grid = new_grid
               
        return new_grid

    def shift_cells_right(self, change_grid_state = True):
        '''Shifts cells right by first flipping the grid horizontally, shifting the cells left on the image grid, 
                then flipping the image grid horizontally to get back to the original orientation of the grid. 
                This is equivalent to just shifting the cells right, but is less work to code.
           change_grid_state: states whether the grid state of the Grid object will be changed. 
                If True, returns the new grid state.
                If False, does not change the grid state but returns the hypothetical grid state that
                    would've been generated if change_grid_state was True.
                change_grid_state is True by default'''
        image_grid = np.fliplr(self.grid)
        image_grid = shift_cells_left_helper(image_grid)
        new_grid = np.fliplr(image_grid)
        
        if change_grid_state == True:
            self.old_grid = self.grid
            self.grid = new_grid
        
        return new_grid

    def shift_cells_up(self, change_grid_state = True):
        '''Shifts cells up by first transposing the grid, shifting the cells left on the transpose grid, 
                then transposing the transpose grid back to get back to the original orientation of the grid. 
                This is equivalent to just shifting the cells up, but is less work to code.
           change_grid_state: states whether the grid state of the Grid object will be changed. 
                If True, returns the new grid state.
                If False, does not change the grid state but returns the hypothetical grid state that
                    would've been generated if change_grid_state was True.
                change_grid_state is True by default'''
        transpose_grid = np.transpose(self.grid)
        transpose_grid = shift_cells_left_helper(transpose_grid)
        new_grid = np.transpose(transpose_grid)
        
        if change_grid_state == True:
            self.old_grid = self.grid
            self.grid = new_grid
        
        return new_grid

    def shift_cells_down(self, change_grid_state = True):
        '''Shifts cells down by first transposing the grid, shifting the cells right on the transpose grid 
                (which is just horizontally flipping the transpose grid, performing a left shift, 
                then horizontally flipping the shifted grid back), 
                then transposing the transpose grid back to get back to the original orientation of the grid. 
                This is equivalent to just shifting the cells down, but is less work to code.
           change_grid_state: states whether the grid state of the Grid object will be changed. 
                If True, returns the new grid state.
                If False, does not change the grid state but returns the hypothetical grid state that
                    would've been generated if change_grid_state was True.
                change_grid_state is True by default'''
        transpose_grid = np.transpose(self.grid)
        image_grid = np.fliplr(transpose_grid)
        image_grid = shift_cells_left_helper(image_grid)
        image_grid = np.fliplr(image_grid)
        new_grid = np.transpose(image_grid)
        
        if change_grid_state == True:
            self.old_grid = self.grid
            self.grid = new_grid

        return new_grid

    def free_cells(self):
        """Returns a list of empty cells."""
        return [(row, col)
                for row in range(self.grid.shape[0])
                for col in range(self.grid.shape[1])
                if not self.grid[row][col]]

    def spawn_new(self, change_grid_state = True):
        '''Spawns one new number in the grid. 90% chance that the new number is a 2,
            and 10% chance that the new number is a 4.'''
        # Gets a list of the coordinates of the free cells in the grid.
        free = self.free_cells()
        
        new_grid = np.copy(self.grid)

        # Randomly picks one free cell, and assigns its value either a 2 or 4.
        #  The chance that the number is a 2 is 90% and the chance that the number is a 4 is 10%.
        row, col = random.sample(free, 1)[0]
        new_grid[row][col] = random.randint(0, 10) and 2 or 4

        if change_grid_state == True:
            self.grid = new_grid
        return new_grid