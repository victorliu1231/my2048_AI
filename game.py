from turtle import st
from Grid import Grid, is_equal_arrays, score, assign_score, apply_two_param_func_to_list_of_2D_arrays, apply_one_param_func_to_list_of_2D_arrays
import numpy as np
import pygame
import sys
import random
from AI import AI

# initialising pygame
pygame.init()
 
# Instantiating the grid
grid_width = 4 #number of columns in the grid
grid_height = 4 #number of rows in the grid

if (grid_width*grid_height < 2):
        raise Exception("Your grid must have more than 1 tile in it.")

starting_grid = np.zeros((grid_height,grid_width))
counter = 0

while counter < 2:
        randrow = random.randint(0, grid_height-1)
        randcol = random.randint(0, grid_width-1)
        if starting_grid[randrow][randcol] != 2:
                starting_grid[randrow][randcol] = 2
                counter = counter + 1

GameGrid = Grid(starting_grid)

# creating display
CELL_WIDTH = 108
CELL_HEIGHT = 108
screen_width = 20 + 10*(grid_width-1) + grid_width*CELL_WIDTH
screen_height = 20 + 10*(grid_height-1) + grid_height*CELL_HEIGHT
BACKGROUND = (0xbb, 0xad, 0xa0)
display = pygame.display.set_mode((screen_width, screen_height))
display.fill(BACKGROUND)

# Populating the screen with the grid cells
FONT = pygame.font.Font('ClearSans-Bold.ttf', 50)
text_2 = FONT.render('2', True, (119, 110, 101))

DEFAULT_TILES = {
        0: [(204, 191, 180), (119, 110, 101)],
        2: [(238, 228, 218), (119, 110, 101)],
        4: [(237, 224, 200), (119, 110, 101)],
        8: [(242, 177, 121), (249, 246, 242)],
        16: [(245, 149, 99), (249, 246, 242)],
        32: [(246, 124, 95), (249, 246, 242)],
        64: [(246, 94, 59), (249, 246, 242)],
        128: [(237, 207, 114), (249, 246, 242)],
        256: [(237, 204, 97), (249, 246, 242)],
        512: [(237, 200, 80), (249, 246, 242)],
        1024: [(237, 197, 63), (249, 246, 242)],
        2048: [(237, 194, 46), (249, 246, 242)],
        4096: [(237, 194, 29), (249, 246, 242)],
        8192: [(237, 194, 12), (249, 246, 242)],
        16384: [(94, 94, 178), (249, 246, 242)],
        32768: [(94, 94, 211), (249, 246, 242)],
        65536: [(94, 94, 233), (249, 246, 242)],
        131072: [(94, 94, 255), (249, 246, 242)],
}

def update_screen(grid):
        for i in range(grid_width):
                for j in range(grid_height):
                        rect_left_side_coord = 10*(i+1) + i*CELL_WIDTH
                        rect_top_side_coord = 10*(j+1) + j*CELL_HEIGHT
                        pygame.draw.rect(display, DEFAULT_TILES[grid[j][i]][0], pygame.Rect(rect_left_side_coord, rect_top_side_coord, CELL_WIDTH, CELL_HEIGHT))
                        if grid[j][i] != 0:
                                text_num = FONT.render(str(int(grid[j][i])), True, DEFAULT_TILES[grid[j][i]][1])
                                text_num_RECT = text_num.get_rect()
                                text_num_RECT.center = (rect_left_side_coord + CELL_WIDTH/2, rect_top_side_coord + CELL_HEIGHT/2)
                                display.blit(text_num, text_num_RECT)
        pygame.display.flip()

def is_any_more_moves(Grid_object):
        is_at_least_one_move = False
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

        for key in predicted_grids.keys():
                if not is_equal_arrays(Grid_object.grid, predicted_grids[key]):
                        is_at_least_one_move = True
        return is_at_least_one_move

#def search(step, grid):
#        if step == 2:
#                return fitness(grid)
#        else:
#                return search(step+1, grid)

update_screen(starting_grid)


# setup for game

print("BEGINNING GRID")
print(GameGrid.grid)
is_AI_on = True
if is_AI_on:
        AI = AI()
        AI_self_run = True # Modify this variable to determine if the AI should run by itself or merely predict the best move
        AI_upon_click = False # Modify this variable to determine if the AI should progress to the next board position only upon clicking the screen

# creating a running loop
while True:
       
    # creating a loop to check events that are occuring
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                if not is_any_more_moves(GameGrid):
                        print("There are no more moves. Try again!")
                        print(GameGrid.grid)
                        pygame.quit()
                        sys.exit()

                if AI_self_run:
                        max_num = np.amax(GameGrid.grid)
                        grids = [GameGrid.shift_cells_left(change_grid_state=False),
                                        GameGrid.shift_cells_right(change_grid_state=False),
                                        GameGrid.shift_cells_up(change_grid_state=False),
                                        GameGrid.shift_cells_down(change_grid_state=False)
                        ]
                        scores = np.array([])
                        for i in range(4):
                                scores = np.append(scores, assign_score(grids[i]))
                        # If the new grid is the same as the old grid, then assign the score of the new grid 
                        #    as 0 to remove the new grid from the selection pool.
                        list_is_new_grid_same_as_old_grid = apply_two_param_func_to_list_of_2D_arrays(grids, GameGrid.grid, is_equal_arrays)
                        for i in range(len(list_is_new_grid_same_as_old_grid)):
                                if list_is_new_grid_same_as_old_grid[i]:
                                        scores[i] = 0
                        highest_score_directions = np.where(scores == np.max(scores))[0]
                        original_highest_score_directions = np.copy(highest_score_directions)
                        # Favors directions that will keep the max tile to the corner.
                        if GameGrid.grid[0][0] == max_num: 
                                highest_score_directions = highest_score_directions[highest_score_directions != 1]
                                highest_score_directions = highest_score_directions[highest_score_directions != 3]
                        if GameGrid.grid[0][-1] == max_num: 
                                highest_score_directions = highest_score_directions[highest_score_directions != 0]
                                highest_score_directions = highest_score_directions[highest_score_directions != 3]
                        if GameGrid.grid[-1][0] == max_num: 
                                highest_score_directions = highest_score_directions[highest_score_directions != 1]
                                highest_score_directions = highest_score_directions[highest_score_directions != 2]
                        if GameGrid.grid[-1][-1] == max_num: 
                                highest_score_directions = highest_score_directions[highest_score_directions != 0]
                                highest_score_directions = highest_score_directions[highest_score_directions != 2]
                        if len(highest_score_directions) == 0:
                                highest_score_directions = original_highest_score_directions
                        # If there is still more than one optimal direction, choose the direction randomly.
                        if len(highest_score_directions) > 1:
                                highest_score_direction = random.sample(list(highest_score_directions), 1)[0]
                        else:
                                highest_score_direction = highest_score_directions[0]
                        print(scores[highest_score_direction])
                        AI.execute_move(highest_score_direction, GameGrid)
                        GameGrid.spawn_new()
                        update_screen(GameGrid.grid)
                        '''triangulation_moves = AI.check_triangulate(GameGrid)
                        if len(triangulation_moves) == 3:
                                print("triangulated")
                                for direction in triangulation_moves:
                                        print(direction)
                                        AI.execute_move(direction, GameGrid)
                                        GameGrid.spawn_new()
                                        update_screen(GameGrid.grid)
                        else:
                                best_direction = AI.predict_best_direction(GameGrid)
                                print(best_direction)
                                AI.execute_move(best_direction, GameGrid)
                                GameGrid.spawn_new()
                                update_screen(GameGrid.grid)'''
                elif AI_upon_click:
                        if event.type == pygame.MOUSEBUTTONUP:
                                max_num = np.amax(GameGrid.grid)
                                grids = [GameGrid.shift_cells_left(change_grid_state=False),
                                         GameGrid.shift_cells_right(change_grid_state=False),
                                         GameGrid.shift_cells_up(change_grid_state=False),
                                         GameGrid.shift_cells_down(change_grid_state=False)
                                ]
                                scores = np.array([])
                                for i in range(4):
                                        scores = np.append(scores, assign_score(grids[i]))
                                # If the new grid is the same as the old grid, then remove the predicted grid from the selection pool.
                                filtered_scores = scores[~apply_two_param_func_to_list_of_2D_arrays(grids, GameGrid.grid, is_equal_arrays)]
                                highest_score_direction = np.where(scores == np.max(filtered_scores))[0]
                                print(highest_score_direction)
                                '''highest_score_directions = np.where(scores == np.max(filtered_scores))[0]
                                original_highest_score_directions = np.copy(highest_score_directions)
                                # Favors directions that will keep the max tile to the corner.
                                if GameGrid.grid[0][0] == max_num: 
                                        highest_score_directions = highest_score_directions[highest_score_directions != 1]
                                        highest_score_directions = highest_score_directions[highest_score_directions != 3]
                                if GameGrid.grid[0][-1] == max_num: 
                                        highest_score_directions = highest_score_directions[highest_score_directions != 0]
                                        highest_score_directions = highest_score_directions[highest_score_directions != 3]
                                if GameGrid.grid[-1][0] == max_num: 
                                        highest_score_directions = highest_score_directions[highest_score_directions != 1]
                                        highest_score_directions = highest_score_directions[highest_score_directions != 2]
                                if GameGrid.grid[-1][-1] == max_num: 
                                        highest_score_directions = highest_score_directions[highest_score_directions != 0]
                                        highest_score_directions = highest_score_directions[highest_score_directions != 2]
                                if len(highest_score_directions) == 0:
                                        highest_score_directions = original_highest_score_directions
                                '''
                                if len(highest_score_direction) > 1:
                                        highest_score_direction = random.sample(list(highest_score_direction), 1)[0]
                                else:
                                        highest_score_direction = highest_score_direction[0]
                                print(scores[highest_score_direction])
                                print(highest_score_direction)
                                AI.execute_move(highest_score_direction, GameGrid)
                                GameGrid.spawn_new()
                                update_screen(GameGrid.grid)
                                '''triangulation_moves = AI.check_triangulate(GameGrid)
                                if len(triangulation_moves) == 3:
                                        for direction in triangulation_moves:
                                                print(direction)
                                                AI.execute_move(direction, GameGrid)
                                                GameGrid.spawn_new()
                                                update_screen(GameGrid.grid)
                                else:
                                        best_direction = AI.predict_best_direction(GameGrid)
                                        print(best_direction)
                                        AI.execute_move(best_direction, GameGrid)
                                        GameGrid.spawn_new()
                                        update_screen(GameGrid.grid)'''
                else:
                        # checking if keydown event happened or not
                        if event.type == pygame.KEYDOWN:
                                # checks if left, right, up, or down arrows were pressed
                                if event.key == pygame.K_UP:
                                        #print("UP")
                                        GameGrid.shift_cells_up()
                                        if not (is_equal_arrays(GameGrid.grid, GameGrid.old_grid)):
                                                GameGrid.spawn_new()
                                                update_screen(GameGrid.grid)
                                                if is_AI_on:
                                                        best_direction = AI.predict_best_direction(GameGrid)
                                                        print(f"Best move: {best_direction}")
                                if event.key == pygame.K_DOWN:
                                        #print("DOWN")
                                        GameGrid.shift_cells_down()
                                        if not (is_equal_arrays(GameGrid.grid, GameGrid.old_grid)):
                                                GameGrid.spawn_new()
                                                update_screen(GameGrid.grid)
                                                if is_AI_on:
                                                        best_direction = AI.predict_best_direction(GameGrid)
                                                        print(f"Best move: {best_direction}")
                                if event.key == pygame.K_LEFT:
                                        #print("LEFT")
                                        GameGrid.shift_cells_left()
                                        if not (is_equal_arrays(GameGrid.grid, GameGrid.old_grid)):
                                                GameGrid.spawn_new()
                                                update_screen(GameGrid.grid)
                                                if is_AI_on:
                                                        best_direction = AI.predict_best_direction(GameGrid)
                                                        print(f"Best move: {best_direction}")
                                if event.key == pygame.K_RIGHT:
                                        #print("RIGHT")
                                        GameGrid.shift_cells_right()
                                        if not (is_equal_arrays(GameGrid.grid, GameGrid.old_grid)):
                                                GameGrid.spawn_new()
                                                update_screen(GameGrid.grid)
                                                if is_AI_on:
                                                        best_direction = AI.predict_best_direction(GameGrid)
                                                        print(f"Best move: {best_direction}")

