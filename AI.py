from decimal import InvalidContext
from operator import index
from tkinter import Grid
import numpy as np
import random
from Grid import Grid, num_adj_tiles, num_adj_tiles_of_number, num_free_cells, is_equal_arrays, is_all_arrays_equal, num_inst_in_grid, kth_max, is_max_tile_in_corner, num_number_adj_to_tile, tile_coords_of_number_along_edge, num_edges_from_a_corner_max_tile_in_sorted_order, apply_one_param_func_to_list_of_2D_arrays, apply_two_param_func_to_list_of_2D_arrays, apply_three_param_func_to_list_of_2D_arrays

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
            print("new max tile")
        
        # If there is still more than one direction in the selection pool, put any predicted grids
        #  where a max tile is in the corner of the grid into the next round of the selection pool.
        if len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
            list_is_max_tile_in_corner = apply_two_param_func_to_list_of_2D_arrays(selection_pool_grids, "boolean", is_max_tile_in_corner)
            index_direction = np.where(list_is_max_tile_in_corner)[0]
            if len(index_direction) == 0:
                index_direction = original_index_direction
            else:
                index_direction = original_index_direction[index_direction]
                print("max tile in corner")

        # If there is a max tile in a corner, eliminate any directions that will move the max tile away from the corner. 
        # Want to incentivize most max tiles when the direction is moving towards the corner max tile and incentivize
        #  most adjacents if the direction is moving away from the corner max tile.
        if len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            most_max_tiles_index_direction = np.copy(index_direction) # variable that will change
            most_adjacent_tiles_index_direction = np.copy(index_direction) # variable that will change
            max_num = np.amax(Grid_object.grid)
            
            # Most max-valued tiles algorithm
            k = 1
            max_valued_successful = False
            while len(most_max_tiles_index_direction) > 1:
                most_max_tiles_original_index_direction = np.copy(most_max_tiles_index_direction)
                selection_pool_grids = numpy_form_of_predicted_grids[most_max_tiles_index_direction]
                kth_largest = np.unique(selection_pool_grids)[-1*k]
                if kth_largest == 0:
                    break
                list_num_kth_largests = apply_two_param_func_to_list_of_2D_arrays(selection_pool_grids, kth_largest, num_inst_in_grid)
                most_max_tiles_index_direction = np.where(list_num_kth_largests == np.max(list_num_kth_largests))[0]
                most_max_tiles_index_direction = most_max_tiles_original_index_direction[most_max_tiles_index_direction]
                k = k + 1
            
            if Grid_object.grid[0][0] == max_num:
                # If moving left or up creates more max_tiles, then only consider moving left or up.
                #   Otherwise, keep the original directions.
                if 0 in most_max_tiles_index_direction or 2 in most_max_tiles_index_direction:
                    index_direction = index_direction[index_direction != 1]
                    index_direction = index_direction[index_direction != 3]
                    max_valued_successful = True
            if Grid_object.grid[0][-1] == max_num:
                # If moving right or up creates more max_tiles, then only consider moving right or up.
                #   Otherwise, keep the original directions.
                if 1 in most_max_tiles_index_direction or 2 in most_max_tiles_index_direction:
                    index_direction = index_direction[index_direction != 0]
                    index_direction = index_direction[index_direction != 3]
                    max_valued_successful = True
            if Grid_object.grid[-1][0] == max_num:
                # If moving left or down creates more max_tiles, then only consider moving left or down.
                #   Otherwise, keep the original directions.
                if 0 in most_max_tiles_index_direction or 3 in most_max_tiles_index_direction:
                    index_direction = index_direction[index_direction != 1]
                    index_direction = index_direction[index_direction != 2]
                    max_valued_successful = True
            if Grid_object.grid[-1][-1] == max_num:
                # If moving right or down creates more max_tiles, then only consider moving right or down.
                #   Otherwise, keep the original directions.
                if 1 in most_max_tiles_index_direction or 3 in most_max_tiles_index_direction:
                    index_direction = index_direction[index_direction != 0]
                    index_direction = index_direction[index_direction != 2]
                    max_valued_successful = True
            
            # Most adjacents algorithm triggers only if max_valued algorithm successfully chose a direction
            if not max_valued_successful:
                n = 1
                while len(most_adjacent_tiles_index_direction) > 1:
                    most_adjacent_tiles_original_index_direction = np.copy(most_adjacent_tiles_index_direction)
                    selection_pool_grids = numpy_form_of_predicted_grids[most_adjacent_tiles_index_direction]
                    uniques = np.unique(selection_pool_grids)
                    if n > len(uniques):
                        break
                    nth_largest_num = uniques[-1*n]
                    if nth_largest_num == 0:
                        break
                    list_num_adj_tiles_of_num = apply_two_param_func_to_list_of_2D_arrays(selection_pool_grids, nth_largest_num, num_adj_tiles_of_number)
                    most_adjacent_tiles_index_direction = np.where(list_num_adj_tiles_of_num == np.max(list_num_adj_tiles_of_num))
                    most_adjacent_tiles_index_direction = most_adjacent_tiles_original_index_direction[most_adjacent_tiles_index_direction]
                    n = n + 1
                
                if len(most_adjacent_tiles_index_direction) != 4:
                    if Grid_object.grid[0][0] == max_num: 
                        # If moving right or down creates more adjacent tiles, then only consider moving right or down.
                        #   Otherwise, keep the original directions.
                        if 1 in most_adjacent_tiles_index_direction or 3 in most_adjacent_tiles_index_direction:
                            index_direction = index_direction[index_direction != 0]
                            index_direction = index_direction[index_direction != 2]
                    if Grid_object.grid[0][-1] == max_num: 
                        # If moving left or down creates more adjacent tiles, then only consider moving left or down.
                        #   Otherwise, keep the original directions.
                        if 0 in most_adjacent_tiles_index_direction or 3 in most_adjacent_tiles_index_direction:
                            index_direction = index_direction[index_direction != 1]
                            index_direction = index_direction[index_direction != 2]
                    if Grid_object.grid[-1][0] == max_num: 
                        # If moving right or up creates more adjacent tiles, then only consider moving right or up.
                        #   Otherwise, keep the original directions.
                        if 1 in most_adjacent_tiles_index_direction or 2 in most_adjacent_tiles_index_direction:
                            index_direction = index_direction[index_direction != 0]
                            index_direction = index_direction[index_direction != 3]
                    if Grid_object.grid[-1][-1] == max_num: 
                        # If moving left or up creates more adjacent tiles, then only consider moving left or up.
                        #   Otherwise, keep the original directions.
                        if 0 in most_adjacent_tiles_index_direction or 2 in most_adjacent_tiles_index_direction:
                            index_direction = index_direction[index_direction != 1]
                            index_direction = index_direction[index_direction != 3]

        # If there is still more than one direction in the selection pool, put any predicted grids
        #  where the number of sorted edges emanating from a max corner tile (if a max corner tile exists)
        #  into the next round of the selection pool.
        if len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
            max_num = np.unique(selection_pool_grids)[-1]
            list_coords_of_max_num = apply_two_param_func_to_list_of_2D_arrays(selection_pool_grids, max_num, tile_coords_of_number_along_edge)
            list_num_sorted_edges = []
            for i in range(len(selection_pool_grids)):
                list_num_sorted_edges.append(num_edges_from_a_corner_max_tile_in_sorted_order(selection_pool_grids[i], list_coords_of_max_num[i]))
            index_direction = np.where(list_num_sorted_edges == np.max(list_num_sorted_edges))
            index_direction = original_index_direction[index_direction]
            print("number of sorted edges")

        # The below algorithm makes the AI favor boards with the largest tile in the corner, the second largest tile adjacent to the largest tile,
        #  the third largest tile adjacent to the second largest tile, and so on.
        '''m = 1
        while len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
            uniques = np.unique(selection_pool_grids)
            if m > len(uniques):
                break
            mth_max_num = uniques[-1*m]
            if mth_max_num == 2 or mth_max_num == 0 or len(index_direction) == 1:
                break
            m_plus_one_th_max_num = uniques[-1*(m+1)]
            list_mth_tile_coords_along_edge_for_all_grids = apply_two_param_func_to_list_of_2D_arrays(selection_pool_grids, mth_max_num, tile_coords_of_number_along_edge)
            list_total_num_mth_tile_adj_to_mth_tile_or_m_plus_one_th_tile_in_all_grids = []
            for i in range(len(selection_pool_grids)):
                selectable_grid = selection_pool_grids[i]
                mth_tile_coords_for_this_grid = list_mth_tile_coords_along_edge_for_all_grids[i]
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
            index_direction = original_index_direction

        # If there is still more than one grid in the selection pool, put the grids with the most
        #   adjacent max tiles in the next round of the selection pool. If there is more than one
        #   grid with the most adjacent max tiles, then put the grids with the most adjacent second-largest
        #   tiles in the next round of the selection pool. If there is still more than one grid in the
        #   selection pool, select the grids with the most adjacent third-largest tiles, and so on.
        n = 1
        while len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
            uniques = np.unique(selection_pool_grids)
            if n > len(uniques):
                break
            nth_largest_num = uniques[-1*n]
            if nth_largest_num == 0:
                break
            list_num_adj_tiles_of_num = apply_two_param_func_to_list_of_2D_arrays(selection_pool_grids, nth_largest_num, num_adj_tiles_of_number)
            index_direction = np.where(list_num_adj_tiles_of_num == np.max(list_num_adj_tiles_of_num))
            index_direction = original_index_direction[index_direction]
            n = n + 1
        if len(index_direction) == 0:
            index_direction = original_index_direction
        else:
            print("most adjacents")

        # If there is still more than one grid in the selection pool, put the grids with the most largest-valued tiles
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
            k = k + 1  

        # If there is still more than one grid in the selection pool, put the grids with the most number of free cells
        #   into the next round of the selection pool.
        if len(index_direction) > 1:
            original_index_direction = np.copy(index_direction)
            selection_pool_grids = numpy_form_of_predicted_grids[index_direction]
            list_num_free_cells = apply_one_param_func_to_list_of_2D_arrays(selection_pool_grids, num_free_cells)
            index_direction = np.where(list_num_free_cells == np.max(list_num_free_cells))[0]
            index_direction = original_index_direction[index_direction]            
        '''

        # If there is still more than one optimal predicted grid, the AI will just choose randomly which
        #   direction to select amongst the optimal predicted grids.
        if len(index_direction) > 1:
            index_direction = random.sample(list(index_direction), 1)[0]
            print("random")
        

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
        if direction == "left" or direction == 0:
            print(Grid_object.shift_cells_left())
        elif direction == "right" or direction == 1:
            print(Grid_object.shift_cells_right())
        elif direction == "up" or direction == 2:
            print(Grid_object.shift_cells_up())
        elif direction == "down" or direction == 3:
            print(Grid_object.shift_cells_down())
        else:
            raise Exception("Keyword Error: direction must either be 'left', 'right', 'up', 'down', 0, 1, 2, or 3.")

    def check_triangulate(self, Grid_object):
        current_max = np.amax(Grid_object.grid)
        triangulation_moves = [] #Instantiation
        index_triangulation = -1 #Instantiation
        # If the max_num is in top left corner and is surrounded by tiles half its value, then see if you
        #  can move the cells to the right, up, then left OR move the cells down, left, then up 
        #  to make a new max value. If so, then make those moves.
        if Grid_object.grid[0][0] == current_max and Grid_object.grid[1][0] == current_max/2 and Grid_object.grid[0][1] == current_max/2: 
            # Counterclockwise triangulation
            new_max_with_counterclockwise = False
            rightShiftedGrid = Grid(Grid_object.shift_cells_right(change_grid_state = False))
            if rightShiftedGrid.grid[0][0] != 0:
                upShiftedGrid = Grid(rightShiftedGrid.shift_cells_up(change_grid_state = False))
                counterclockwiseTriangulatedGrid = Grid(upShiftedGrid.shift_cells_left(change_grid_state = False))
                max_triangulated_grid = np.amax(counterclockwiseTriangulatedGrid.grid)
                if max_triangulated_grid > current_max:
                    new_max_with_counterclockwise = True
                    index_triangulation = 0
                    #[1, 2, 0] right, up, then left
            # Clockwise triangulation
            new_max_with_clockwise = False
            downShiftedGrid = Grid(Grid_object.shift_cells_down(change_grid_state = False))
            if downShiftedGrid.grid[0][0] != 0:
                leftShiftedGrid = Grid(downShiftedGrid.shift_cells_left(change_grid_state = False))
                clockwiseTriangulatedGrid = Grid(leftShiftedGrid.shift_cells_up(change_grid_state = False))
                max_triangulated_grid = np.amax(clockwiseTriangulatedGrid.grid)
                if max_triangulated_grid > current_max:
                    new_max_with_clockwise = True
                    index_triangulation = 1
                    #[3, 0, 2] down, left, then up
            if new_max_with_counterclockwise and new_max_with_clockwise:
                list_num_sorted_edges = apply_two_param_func_to_list_of_2D_arrays([counterclockwiseTriangulatedGrid, clockwiseTriangulatedGrid], [0, 0], num_edges_from_a_corner_max_tile_in_sorted_order)
                index_triangulation = np.where(list_num_sorted_edges == np.max(list_num_sorted_edges))
            # If there more than one optimal triangulation direction, the AI will just choose randomly which
            #   triangulation direction to choose.
            if type(index_triangulation) == np.ndarray and len(index_triangulation) > 1:
                index_triangulation = random.sample(list(index_triangulation), 1)[0]
            if index_triangulation == 0:
                triangulation_moves = [1, 2, 0]
            elif index_triangulation == 1:
                triangulation_moves = [3, 0, 2]

        # If the max_num is in top right corner and is surrounded by tiles half its value, then see if you
        #  can move the cells to the down, right, then up OR move the cells left, up, then right
        #  to make a new max value. If so, then make those moves.
        if Grid_object.grid[0][-1] == current_max and Grid_object.grid[1][-1] == current_max/2 and Grid_object.grid[0][-2] == current_max/2: 
            # Counterclockwise triangulation
            new_max_with_counterclockwise = False
            downShiftedGrid = Grid(Grid_object.shift_cells_down(change_grid_state = False))
            if downShiftedGrid.grid[0][-1] != 0:
                rightShiftedGrid = Grid(downShiftedGrid.shift_cells_right(change_grid_state = False))
                counterclockwiseTriangulatedGrid = Grid(rightShiftedGrid.shift_cells_up(change_grid_state = False))
                max_triangulated_grid = np.amax(counterclockwiseTriangulatedGrid.grid)
                if max_triangulated_grid > current_max:
                    new_max_with_counterclockwise = True
                    index_triangulation = 0
                    #[3, 1, 2] down, right, then up
            # Clockwise triangulation
            new_max_with_clockwise = False
            leftShiftedGrid = Grid(Grid_object.shift_cells_left(change_grid_state = False))
            if leftShiftedGrid.grid[0][0] != 0:
                upShiftedGrid = Grid(leftShiftedGrid.shift_cells_up(change_grid_state = False))
                clockwiseTriangulatedGrid = Grid(upShiftedGrid.shift_cells_right(change_grid_state = False))
                max_triangulated_grid = np.amax(clockwiseTriangulatedGrid.grid)
                if max_triangulated_grid > current_max:
                    new_max_with_clockwise = True
                    index_triangulation = 1
                    #[0, 2, 1] left, up, then right
            if new_max_with_counterclockwise and new_max_with_clockwise:
                list_num_sorted_edges = apply_two_param_func_to_list_of_2D_arrays([counterclockwiseTriangulatedGrid, clockwiseTriangulatedGrid], [0, -1], num_edges_from_a_corner_max_tile_in_sorted_order)
                index_triangulation = np.where(list_num_sorted_edges == np.max(list_num_sorted_edges))
            # If there more than one optimal triangulation direction, the AI will just choose randomly which
            #   triangulation direction to choose.
            if type(index_triangulation) == np.ndarray and len(index_triangulation) > 1:
                index_triangulation = random.sample(list(index_triangulation), 1)[0]
            if index_triangulation == 0:
                triangulation_moves = [3, 1, 2]
            elif index_triangulation == 1:
                triangulation_moves = [0, 2, 1]

        # If the max_num is in bottom left corner and is surrounded by tiles half its value, then see if you
        #  can move the cells to the up, left, then down OR move the cells right, down, then left
        #  to make a new max value. If so, then make those moves.
        if Grid_object.grid[-1][0] == current_max and Grid_object.grid[-2][0] == current_max/2 and Grid_object.grid[-1][1] == current_max/2: 
            # Counterclockwise triangulation
            new_max_with_counterclockwise = False
            upShiftedGrid = Grid(Grid_object.shift_cells_up(change_grid_state = False))
            if upShiftedGrid.grid[-1][0] != 0:
                leftShiftedGrid = Grid(upShiftedGrid.shift_cells_left(change_grid_state = False))
                counterclockwiseTriangulatedGrid = Grid(leftShiftedGrid.shift_cells_down(change_grid_state = False))
                max_triangulated_grid = np.amax(counterclockwiseTriangulatedGrid.grid)
                if max_triangulated_grid > current_max:
                    new_max_with_counterclockwise = True
                    index_triangulation = 0
                    #[2, 0, 3] up, left, then down
            # Clockwise triangulation
            new_max_with_clockwise = False
            rightShiftedGrid = Grid(Grid_object.shift_cells_right(change_grid_state = False))
            if rightShiftedGrid.grid[-1][0] != 0:
                downShiftedGrid = Grid(rightShiftedGrid.shift_cells_down(change_grid_state = False))
                clockwiseTriangulatedGrid = Grid(downShiftedGrid.shift_cells_left(change_grid_state = False))
                max_triangulated_grid = np.amax(clockwiseTriangulatedGrid.grid)
                if max_triangulated_grid > current_max:
                    new_max_with_clockwise = True
                    index_triangulation = 1
                    #[1, 3, 0] right, down, then left
            if new_max_with_counterclockwise and new_max_with_clockwise:
                list_num_sorted_edges = apply_two_param_func_to_list_of_2D_arrays([counterclockwiseTriangulatedGrid, clockwiseTriangulatedGrid], [-1, 0], num_edges_from_a_corner_max_tile_in_sorted_order)
                index_triangulation = np.where(list_num_sorted_edges == np.max(list_num_sorted_edges))
            # If there more than one optimal triangulation direction, the AI will just choose randomly which
            #   triangulation direction to choose.
            if type(index_triangulation) == np.ndarray and len(index_triangulation) > 1:
                index_triangulation = random.sample(list(index_triangulation), 1)[0]
            if index_triangulation == 0:
                triangulation_moves = [2, 0, 3]
            elif index_triangulation == 1:
                triangulation_moves = [1, 3, 0]
        
        # If the max_num is in bottom right corner and is surrounded by tiles half its value, then see if you
        #  can move the cells to the left, down, then right OR move the cells up, right, then down
        #  to make a new max value. If so, then make those moves.
        if Grid_object.grid[-1][-1] == current_max and Grid_object.grid[-2][-1] == current_max/2 and Grid_object.grid[-1][-2] == current_max/2: 
            # Counterclockwise triangulation
            new_max_with_counterclockwise = False
            leftShiftedGrid = Grid(Grid_object.shift_cells_left(change_grid_state = False))
            if leftShiftedGrid.grid[-1][-1] != 0:
                downShiftedGrid = Grid(leftShiftedGrid.shift_cells_down(change_grid_state = False))
                counterclockwiseTriangulatedGrid = Grid(downShiftedGrid.shift_cells_right(change_grid_state = False))
                max_triangulated_grid = np.amax(counterclockwiseTriangulatedGrid.grid)
                if max_triangulated_grid > current_max:
                    new_max_with_counterclockwise = True
                    index_triangulation = 0
                    #[0, 3, 1] left, down, then right
            # Clockwise triangulation
            new_max_with_clockwise = False
            upShiftedGrid = Grid(Grid_object.shift_cells_up(change_grid_state = False))
            if upShiftedGrid.grid[-1][-1] != 0:
                rightShiftedGrid = Grid(upShiftedGrid.shift_cells_right(change_grid_state = False))
                clockwiseTriangulatedGrid = Grid(rightShiftedGrid.shift_cells_down(change_grid_state = False))
                max_triangulated_grid = np.amax(clockwiseTriangulatedGrid.grid)
                if max_triangulated_grid > current_max:
                    new_max_with_clockwise = True
                    index_triangulation = 1
                    #[2, 1, 3] up, right, then down
            if new_max_with_counterclockwise and new_max_with_clockwise:
                list_num_sorted_edges = apply_two_param_func_to_list_of_2D_arrays([counterclockwiseTriangulatedGrid, clockwiseTriangulatedGrid], [-1, -1], num_edges_from_a_corner_max_tile_in_sorted_order)
                index_triangulation = np.where(list_num_sorted_edges == np.max(list_num_sorted_edges))
            # If there more than one optimal triangulation direction, the AI will just choose randomly which
            #   triangulation direction to choose.
            if type(index_triangulation) == np.ndarray and len(index_triangulation) > 1:
                index_triangulation = random.sample(list(index_triangulation), 1)[0]
            if index_triangulation == 0:
                triangulation_moves = [0, 3, 1]
            elif index_triangulation == 1:
                triangulation_moves = [2, 1, 3]
        return triangulation_moves
        