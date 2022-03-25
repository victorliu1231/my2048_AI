from operator import index
from tkinter import Grid
import numpy as np
import random
from Grid import num_adj_tiles, num_free_cells, is_equal_arrays, is_all_arrays_equal, num_inst_in_grid, kth_max, is_max_tile_in_corner, num_number_adj_to_tile, tile_coords_of_number, apply_one_param_func_to_list_of_2D_arrays, apply_two_param_func_to_list_of_2D_arrays, apply_three_param_func_to_list_of_2D_arrays

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

        numpy_form_of_predicted_grids = np.array(list(predicted_grids.values()))

        # Remove any predicted grids that are the exact same as the current grid state from the selection pool.
        #  This is because if the predicted grid is the exact same as the current grid state, that means the
        #  the move that would've generated that predicted grid is invalid.
        index_direction = np.where(~apply_two_param_func_to_list_of_2D_arrays(numpy_form_of_predicted_grids, Grid_object.grid, is_equal_arrays))[0]

        # Puts any directions that will create a new "max" tile into the selection pool.
        selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
        max_num = np.unique(selection_pool_grids)[-1]
        if max_num > np.amax(Grid_object.grid):
            new_index_direction = []
            for i in range(len(selection_pool_grids)):
                if max_num in selection_pool_grids[i]:
                    new_index_direction.append(i)
            index_direction = index_direction[new_index_direction]

        # If there is still more than one direction in the selection pool, put any predicted grids
        #  where a max tile is in the corner of the grid into the next round of the selection pool.
        if len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
            list_is_max_tile_in_corner = apply_two_param_func_to_list_of_2D_arrays(selection_pool_grids, "boolean", is_max_tile_in_corner)
            index_direction = np.where(list_is_max_tile_in_corner)[0]
            index_direction = original_index_direction[index_direction]
            # The below algorithm makes the AI favor boards with the largest tile in the corner, the second largest tile adjacent to the largest tile,
            #  the third largest tile adjacent to the second largest tile, and so on.
            m = 1
            while len(index_direction) > 1:
                original_index_direction = np.copy(index_direction)
                selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
                all_possible_tiles = np.unique(selection_pool_grids)
                if m > len(all_possible_tiles):
                    break
                mth_max_num = all_possible_tiles[-1*m]
                if mth_max_num == 2 or mth_max_num == 0 or len(index_direction) == 1:
                    break
                m_plus_one_th_max_num = all_possible_tiles[-1*(m+1)]
                list_mth_tile_coords_for_all_grids = apply_two_param_func_to_list_of_2D_arrays(selection_pool_grids, mth_max_num ,tile_coords_of_number)
                list_total_num_mth_tile_adj_to_mth_tile_or_m_plus_one_th_tile_in_all_grids = []
                for i in range(len(selection_pool_grids)):
                    selectable_grid = selection_pool_grids[i]
                    mth_tile_coords_for_this_grid = list_mth_tile_coords_for_all_grids[i]
                    total_num_mth_tile_adj_to_mth_tile_or_m_plus_one_th_tile_coord_for_this_grid = 0
                    for mth_tile_coord in mth_tile_coords_for_this_grid:
                        num_mth_tile_adj_to_mth_tile_coord = num_number_adj_to_tile(selectable_grid, mth_max_num, mth_tile_coord)
                        num_mth_tile_adj_to_m_plus_one_th_tile_coord = num_number_adj_to_tile(selectable_grid, m_plus_one_th_max_num, mth_tile_coord)
                        total_num_mth_tile_adj_to_mth_tile_or_m_plus_one_th_tile_coord_for_this_grid = total_num_mth_tile_adj_to_mth_tile_or_m_plus_one_th_tile_coord_for_this_grid + num_mth_tile_adj_to_mth_tile_coord + num_mth_tile_adj_to_m_plus_one_th_tile_coord
                    list_total_num_mth_tile_adj_to_mth_tile_or_m_plus_one_th_tile_in_all_grids.append(total_num_mth_tile_adj_to_mth_tile_or_m_plus_one_th_tile_coord_for_this_grid)
                index_direction = np.where(list_total_num_mth_tile_adj_to_mth_tile_or_m_plus_one_th_tile_in_all_grids == np.max(list_total_num_mth_tile_adj_to_mth_tile_or_m_plus_one_th_tile_in_all_grids))[0]
                index_direction = original_index_direction[index_direction]   
                m = m + 1
            if len(index_direction) == 0:
                #print(len(index_direction))
                #print(len(original_index_direction))
                index_direction = original_index_direction
            
            '''if len(index_direction) >= 1:
                index_direction = original_index_direction[index_direction]
                # Put any predicted grids where a max tile is in the corner of the grid AND the corner max tile
                #  is adjacent to another max tile or adjacent to a second largest max tile into the selection pool.
                if len(index_direction) > 1:
                    original_index_direction = np.copy(index_direction)
                    selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
                    max_num = np.unique(selection_pool_grids)[-1]
                    second_max_num = np.unique(selection_pool_grids)[-2]
                    list_is_max_tile_in_corner_and_adj_to_max_tile = apply_three_param_func_to_list_of_2D_arrays(selection_pool_grids, max_num, max_num, is_tile1_and_tile2_adj)
                    list_is_max_tile_in_corner_and_adj_to_second_largest_max_tile = apply_three_param_func_to_list_of_2D_arrays(selection_pool_grids, max_num, second_max_num, is_tile1_and_tile2_adj)
                    list_is_max_tile_in_corner_and_adj_to_max_tile_or_second_largest_max_tile = np.any([list_is_max_tile_in_corner_and_adj_to_max_tile, list_is_max_tile_in_corner_and_adj_to_second_largest_max_tile], axis=0)
                    index_direction = np.where(list_is_max_tile_in_corner_and_adj_to_max_tile_or_second_largest_max_tile)[0]
                    if len(index_direction) >= 1:
                        index_direction = original_index_direction[index_direction]
                        if len(index_direction) > 1:
                            ...
                    else:
                        index_direction = original_index_direction'''

        # If there is still more than one grid in the selection pool, put the grids with the most number of free cells
        #   into the next round of the selection pool.
        if len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
            list_num_free_cells = apply_one_param_func_to_list_of_2D_arrays(selection_pool_grids, num_free_cells)
            index_direction = np.where(list_num_free_cells == np.max(list_num_free_cells))[0]
            index_direction = original_index_direction[index_direction]

        '''# If there is still more than one grid in the selection pool, put the grids with the most largest-valued tiles
        #  into the next round of the selection pool.
        #  If more than one grid has the most largest-valued tiles, then the AI will select which grid has the 
        #  most 2nd largest tiles, then 3rd largest tiles, and so on.
        #  The only way for there to be more than one grid in the selection pool after this round is if
        #  there is at least one pair of duplicate grids.
        k = 1
        while len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
            kth_largest = np.unique(selection_pool_grids)[-1*k]
            if kth_largest == 0:
                break
            list_num_kth_largests = apply_two_param_func_to_list_of_2D_arrays(selection_pool_grids, kth_largest, num_inst_in_grid)
            index_direction = np.where(list_num_kth_largests == np.max(list_num_kth_largests))[0]
            index_direction = original_index_direction[index_direction]
            k = k + 1'''            

        '''# If there is still more than one grid in the selection pool, then the AI will 
        #   select (out of that list) the grid with the most number of adjacent tiles with the same number.
        if len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
            list_num_adj_tiles = apply_one_param_func_to_list_of_2D_arrays(selection_pool_grids, num_adj_tiles)
            index_direction = np.where(list_num_adj_tiles == np.max(list_num_adj_tiles))[0]
            index_direction = original_index_direction[index_direction]'''       

        # If there is still more than one optimal predicted grid, the AI will just choose randomly which
        #   direction to select amongst the optimal predicted grids.
        if len(index_direction) > 1:
            index_direction = random.sample(list(index_direction), 1)[0]
        
        direction = ''
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
