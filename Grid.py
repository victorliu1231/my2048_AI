from cgi import test
from unittest import skip
import numpy as np
import random

def score(grid):
    return np.sum(grid)

def is_equal_arrays(array1, array2):
    return (array1 == array2).all()

def is_all_arrays_equal(list_of_arrs):
    if len(list_of_arrs) == 1:
        return True
    return True if len(np.unique(list_of_arrs, axis=0)) == 1 else False


def remove_inner_zeros(array_1d):
    no_inner_zeroes_array_1d = array_1d[array_1d != 0]
    pad_length = len(array_1d) - len(no_inner_zeroes_array_1d)
    no_inner_zeroes_array_1d = np.pad(no_inner_zeroes_array_1d, (0, pad_length), "constant")
    return no_inner_zeroes_array_1d

def shift_cells_left_helper(grid):
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

def num_adj_tiles(grid):
    '''Counts the total number of pairs of mergeable adjacent tiles in both the side-to-side and up-and-down directions.
        Calculates whether there are more pairs of mergeable adjacent tiles in the side-to-side or up-and-down direction.
        Returns both the total number of pairs of mergeable adjacent tiles in the direction with the most pairs of mergeable adjacent tiles 
            and the direction with the most pairs of mergeable adjacent tiles.'''
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
    #   direction or the up-and-down direction. Returns both the total number of pairs of mergeable adjacent 
    #   tiles in the direction with the most pairs of mergeable adjacent tiles and the direction 
    #   with the most pairs of mergeable adjacent tiles. If the total number of pairs of mergeable adjacent
    #   tiles is the same in both the side-to-side and the up-and-down direction, return both of the directions.
    array_num_adj_tiles_both_dir = np.array([num_side_to_side_adj_tiles, num_up_and_down_adj_tiles])
    max_num_adj_tiles = max(num_side_to_side_adj_tiles, num_up_and_down_adj_tiles)
    dir_of_max_num_adj_tiles = np.where(array_num_adj_tiles_both_dir == max_num_adj_tiles)[0]
    return max_num_adj_tiles, dir_of_max_num_adj_tiles

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