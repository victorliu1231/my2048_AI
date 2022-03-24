from operator import index
from tkinter import Grid
import numpy as np
import random
from Grid import num_adj_tiles, num_free_cells, is_equal_arrays

class AI:
    def __init__(self):
        return

    def predict_best_direction(self, Grid_object):
        '''Predict which move (left, right, up, or down) will produce the highest score'''
        # First get the predicted grids that would occur if you shift cells either left, right, up, or down
        predicted_grid_left = Grid_object.shift_cells_left(change_grid_state = False)
        predicted_grid_right = Grid_object.shift_cells_right(change_grid_state = False)
        predicted_grid_up = Grid_object.shift_cells_up(change_grid_state = False)
        predicted_grid_down = Grid_object.shift_cells_down(change_grid_state = False)
        
        predicted_grids = {
            0: predicted_grid_left,
            1: predicted_grid_right,
            2: predicted_grid_up,
            3: predicted_grid_down
        }

        # Next, the AI will select which predicted grid has the largest number out of all the grids
        predicted_maxs = np.array([])
        for key in predicted_grids.keys():
            if not is_equal_arrays(Grid_object.grid, predicted_grids[key]):
                predicted_maxs = np.append(predicted_maxs, np.max(predicted_grids[key]))
            else:
                predicted_maxs = np.append(predicted_maxs, 0) # If the shift_cell move is invalid, the AI will cull out this move from its selection pool by setting this move's max value to 0
        #print(f"maxs: {predicted_maxs}")
        index_direction = np.where(predicted_maxs == np.max(predicted_maxs))[0]

        # If there is more than one predicted grid with the largest maximum number, then the AI will 
        #   select (out of that list) the grid with the most number of adjacent tiles with the same number.
        if len(index_direction) > 1:
            predicted_num_adj_tiles = np.array([])
            original_index_direction = np.copy(index_direction)
            for index in index_direction:
                predicted_num_adj_tiles = np.append(predicted_num_adj_tiles, num_adj_tiles(predicted_grids[index])[0])
                #print(predicted_grids[index])
            #print(f"num_adj_tiles: {predicted_num_adj_tiles}")
            index_direction = np.where(predicted_num_adj_tiles == np.max(predicted_num_adj_tiles))[0]
            index_direction = original_index_direction[index_direction]
        
        # If there is more than one predicted grid with the largest maximum number AND
        #   there is more than one predicted grid with the most number of adjacent tiles with the same number, 
        #   then the AI will select (out of that list) the grid with the most number of free cells.
        if len(index_direction) > 1:
            predicted_num_free_cells = np.array([])
            original_index_direction = np.copy(index_direction)
            for index in index_direction:
                predicted_num_free_cells = np.append(predicted_num_free_cells, num_free_cells(predicted_grids[index]))
                #print(predicted_grids[index])
            #print(f"num_free_cells: {predicted_num_free_cells}")
            index_direction = np.where(predicted_num_free_cells == np.max(predicted_num_free_cells))[0]
            index_direction = original_index_direction[index_direction]

        # If there is still more than one optimal predicted grid, the AI will just choose randomly which
        #   direction to select amongst the optimal predicted grids.
        if len(index_direction) > 1:
            index_direction = random.sample(list(index_direction), 1)[0]
        
        direction = ""
        if index_direction == 0:
            direction = "left"
        elif index_direction == 1:
            direction = "right"
        elif index_direction == 2:
            direction = "up"
        elif index_direction == 3:
            direction = "down"
        return direction

    def execute_move(self, direction, Grid_object):
        """Shift cells in the given direction"""
        if direction == "left":
            Grid_object.shift_cells_left()
        elif direction == "right":
            Grid_object.shift_cells_right()
        elif direction == "up":
            Grid_object.shift_cells_up()
        elif direction == "down":
            Grid_object.shift_cells_down()
        else:
            raise Exception("Keyword Error: direction must either be 'left', 'right', 'up', or 'down'.")
