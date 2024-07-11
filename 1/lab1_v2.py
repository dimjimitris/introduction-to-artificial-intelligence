import math
from queue import PriorityQueue
import time
import copy
import random
import numpy as np

# ANSI escape code for yellow
YELLOW = "\033[93m"
RESET = "\033[0m"
    
## some heuristic functions
def euclidean(start, finish):
    return math.sqrt((finish[0] - start[0])**2 + (finish[1] - start[1])**2)

def manhattan(start, finish):
    return abs(finish[0] - start[0]) + abs(finish[1] - start[1])

# Returns valid neighbors
def get_neighbors(maze, position):
    x, y = position
    # The four traversable directions (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, 1), (0, -1)]
    neighbors = []
    for d in directions:
        neighbor = (x + d[0], y + d[1])
        x_n, y_n = neighbor
        if 0 <= x_n < len(maze) and 0 <= y_n < len(maze[0]) and not maze[x_n][y_n]:
            neighbors.append(neighbor)
    return neighbors

def greedy(maze, start, finish, heuristic):
    """
    Greedy best-first search

    Parameters:
    - maze: The 2D matrix that represents the maze with 0 represents empty space and 1 represents a wall
    - start: A tuple with the coordinates of starting position
    - finish: A tuple with the coordinates of finishing position
    - heuristic: A function that takes two tuples and returns a number,
      representing the estimated cost to reach the finish from the start

    Returns:
    - Number of steps from start to finish, equals -1 if the path is not found
    - Viz - everything required for step-by-step vizualization
    
    """
    frontier = PriorityQueue()
    frontier.put((0, start))
    predecessor = {}
    explored = set()
    predecessor[start] = None
    no_expanded_nodes = 0
    search_history = [(copy.deepcopy(frontier.queue), copy.deepcopy(explored))]

    while not frontier.empty():
        current = frontier.get()[1]
        if current == finish:
            break

        no_expanded_nodes += 1

        for next in get_neighbors(maze, current):
            if next not in explored and all(next != pos for _, pos in frontier.queue):
                priority = heuristic(next, finish)
                frontier.put((priority, next))
                predecessor[next] = current
        explored.add(current)
        search_history.append((copy.deepcopy(frontier.queue), copy.deepcopy(explored)))

    path = []
    while current != start:
        path.append(current)
        current = predecessor[current]
    path.append(start)
    path.reverse()

    return (
        len(path) - 1 if path[-1] == finish else -1, # -1 if the path is not found
        (maze, path, finish, search_history, no_expanded_nodes), # no_expanded_nodes can be used for analysis
    )

def vizualize(viz):
    """
    Vizualization function. Shows step by step the work of the search algorithm.
    Symbols used in visualization:
        S: start
        F: finish
        #: wall
        .: empty
        *: explored
        o: frontier
        P: maze cell in the final path

    Parameters:
    - viz: everything required for step-by-step vizualization
    """
    maze, path, finish, search_history, _ = viz
    print("Maze:")
    for i, row in enumerate(maze):
            for j, cell in enumerate(row):
                if (i, j) == path[0]:
                    print('S', end = ' ')
                elif (i, j) == finish:
                    print('F', end = ' ')
                elif cell == 1:
                    print('#', end = ' ')
                else:
                    print('.', end = ' ')
            print()
    time.sleep(1.6)
    print('\n' * 5)

    for phase, (frontier, explored) in enumerate(search_history):
        print(f"Phase: {phase}")
        for i, row in enumerate(maze):
            for j, cell in enumerate(row):
                if (i, j) == path[0]:
                    print('S', end = ' ')
                elif (i, j) == finish:
                    print('F', end = ' ')
                elif (i, j) in explored:
                    print('*', end = ' ')
                elif any((i, j) ==  position for _, position in frontier):
                    print('o', end = ' ')
                elif cell == 1:
                    print('#', end = ' ')
                else:
                    print('.', end = ' ')
            print()
        time.sleep(0.2)
        print('\n' * 5)
    
    # Finally, display the final path
    if path[-1] == finish:
        print("Final Path:")
        for i, row in enumerate(maze):
                for j, cell in enumerate(row):
                    if (i, j) in path:
                        if (i, j) == path[0]:
                            print(f'{YELLOW}S{RESET}', end = ' ')
                        elif (i, j) == path[-1]:
                            if (i, j) == finish:
                                print(f'{YELLOW}F{RESET}', end = ' ')
                            else:
                                print(f'*', end = ' ')
                        else:
                            print(f'{YELLOW}P{RESET}', end = ' ')
                    elif cell == 1:
                        print('#', end = ' ')
                    elif (i, j) in explored:
                        print('*', end = ' ')
                    elif any((i, j) ==  position for _, position in frontier):
                        print('o', end = ' ')
                    else:
                        print('.', end = ' ')
                print()
    else:
        print("No path found.")


# Example usage with known mazes:
# mazes that are solvable:
mazes = [
    ([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ] , (0, 0), (4, 4)), # 5x5 maze

    # a maze with dead ends
    ([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ] , (0, 2), (6, 4)), # 7x7 maze

    # a maze with sparse walls
    ([
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0],
    ] , (0, 6), (6, 0)), # 7x7 maze

    # empty maze
    ([
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ] , (1, 0), (6, 5)), # 7x7 maze

    # euclidean performs better
    ([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0],
    ] , (0, 0), (6, 5)), # 7x7 maze

    # manhattan performs better
    ([
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0],
    ] , (0, 0), (6, 5)), # 7x7 maze

    ([
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
    ], (0, 0), (6, 6)), # 7x7 maze

    ([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ], (0, 0), (10, 9)), # 11x11 maze
    ([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    ], (0, 1), (7, 7)), # 13x13 maze
    ([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    ], (0, 0), (14, 12)), # 15x15 maze
]

for maze, start_position, finish_position in mazes:
    for heur_name, heur in [('euclidean', euclidean), ('manhattan', manhattan)]:
        print(f"Using {heur_name} heuristic:")
        num_steps, viz = greedy(maze, start_position, finish_position, heur)

        # Print number of steps in path
        if num_steps != -1:
            print(f"Path from {start_position} to {finish_position} using greedy best-first search is {num_steps} steps.")

        else:
            print(f"No path from {start_position} to {finish_position} exists.")

        # Vizualize algorithm step-by-step even if the path was not found
        vizualize(viz)

        print()

# mazes that are not solvable:
non_solvable_mazes = [
    # a maze of just walls except for the start and finish
    ([
        [0, 1, 1 ,1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
    ], (0, 0), (4, 4)), # 5x5 maze
    ([
        [0, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0],
    ], (0, 0), (6, 6)), # 7x7 maze

    ([
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ], (0, 0), (10, 9)), # 11x11 maze
]

for maze, start_position, finish_position in non_solvable_mazes:
    for heur_name, heur in [('euclidean', euclidean), ('manhattan', manhattan)]:
        print(f"Using {heur_name} heuristic:")
        num_steps, viz = greedy(maze, start_position, finish_position, heur)

        # Print number of steps in path
        if num_steps != -1:
            print(f"Path from {start_position} to {finish_position} using greedy best-first search is {num_steps} steps.")

        else:
            print(f"No path from {start_position} to {finish_position} exists.")

        # Vizualize algorithm step-by-step even if the path was not found
        vizualize(viz)

        print()

# Some comparisons with random mazes
print()

# start position should be on the border of the maze and size should be an odd number...
# To see why, read comments bellow.
def generate_maze(size, start, finish, threshold=0.08):
    """
    size: integer that indicates the size of the NxN grid of the maze
    start: pair of integers that indicates the coordinates of the starting point (S)
    finish: pair of integers that indicates the coordinates of the finish point (F)
    You can add any other parameters you want to customize maze creation (e.g. variables that
    control the creation of additional paths)

    """

    assert size > 2

    # Given a tuple, this algorithm works by finding its neighbors by skipping one cell in each direction. Thus, if
    # the start is on the border of the maze, then the algorithm will never consider turning the cells on some of the borders
    # into empty cells, leading to problems and bugs. To combat this, we add the following assertion.
    assert size % 2 == 1

    ## Initialize grid 

    grid = np.ones((size, size), dtype=bool)

    def neighbors(node, size, visited, threshold):
        """ 
        Returns all neighbors of a node that are either unvisited, or they are visited but
        there is a wall between the node and the neighbor and the neighbor passes a random test.
        """

        l = []
        x, y = node

        # first condition in all checks is for boundaries
        # neighbors are +-2 in x or y
        # walls are +-1

        if x > 1 and (visited[x-2, y] or (visited[x-1,y] and random.uniform(0,1) <= threshold)):
            l.append((x-2, y))
        if x < size-2 and (visited[x+2, y] or (visited[x+1, y] and random.uniform(0,1) <= threshold)):
            l.append((x+2, y))

        if y > 1 and (visited[x, y-2] or (visited[x, y-1] and random.uniform(0,1) <= threshold)):
            l.append((x, y-2))

        if y < size-2 and (visited[x, y+2] or (visited[x, y+1] and random.uniform(0,1) <= threshold)):
            l.append((x, y+2))

        return l

    stack = []
    stack.append(start)
    grid[start] = False

    while stack:
        current_node = stack.pop()
        # get all unvisited neighbors (and some visited ones with a random chance)
        n = neighbors(current_node, size, grid, threshold)
        if len(n):
            stack.append(current_node)

            # select a random neighbor
            next_node = random.choice(n)

            # break the wall between current and next node
            # find wall!
            # the wall is the block between current and next, so we find that
            (x, y) = current_node
            (x_n, y_n) = next_node

            wall = ((x+x_n)//2, (y+y_n)//2)
            
            grid[wall] = False

            # mark next node as visited and add it to the stack
            grid[next_node] = False
            stack.append(next_node)

    grid[start] = False
    grid[finish] = False

    # turn grid to ints
    grid = grid.astype(int)
    
    return grid

random.seed(17) # so that results are reproducible
results = {}

for size in [41, 61, 81, 101]:
    results[size] = {}
    for _ in range(20):
        maze = generate_maze(size, (0, 0), (size-1, size-1), 0.04)
        for heur_name, heur in [('euclidean', euclidean), ('manhattan', manhattan)][:]:
            path_length, (maze, path, finish, search_history, no_expanded_nodes) = greedy(maze, (0, 0), (size-1, size-1), heur)
            if heur_name not in results[size]:
                results[size][heur_name] = (path_length, no_expanded_nodes)
            else:
                pl, no_e_n = results[size][heur_name]
                results[size][heur_name] = (pl + path_length, no_e_n + no_expanded_nodes)
    
    # get an average of the results
    for heur_name in ['euclidean', 'manhattan']:
        results[size][heur_name] = (results[size][heur_name][0] / 20, results[size][heur_name][1] / 20)
    
    print(f"Results for {size}x{size} maze:")
    print(f"Average path length using euclidean heuristic: {results[size]['euclidean'][0]}")
    print(f"Average number of expanded nodes using euclidean heuristic: {results[size]['euclidean'][1]}")
    print(f"Average path length using manhattan heuristic: {results[size]['manhattan'][0]}")
    print(f"Average number of expanded nodes using manhattan heuristic: {results[size]['manhattan'][1]}")
    print()
