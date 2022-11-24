import pygame
import random
import os
import time
import neat
import visualize
import matplotlib.pyplot as plt
import numpy as np
pygame.font.init()  # init font

WIDTH = 600
HEIGHT = 800
FLOOR = 730

BACKGROUND_WIDTH = 600
BACKGROUND_HEIGHT = 900

NUM_OF_GENERATIONS = 50
SCORE_LIMIT = 20

# for the meme of comicsans
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
BIRD_TO_PIPE_LINES = True

WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("445 Project :)")

PLAYER_JUMP_VELOCITY = -10.5
PLAYER_TERMINAL_VELOCITY = 16

pipe_image = pygame.transform.scale2x(pygame.image.load('images/pipe.png').convert_alpha())
background_image = pygame.transform.scale(pygame.image.load('images/bg.png').convert_alpha(), (BACKGROUND_WIDTH, BACKGROUND_HEIGHT))
player_images = []
for x in range(1, 4):
    player_images.append(pygame.transform.scale2x(pygame.image.load('images/bird' + str(x) + '.png')))
ground_image = pygame.transform.scale2x(pygame.image.load('images/base.png').convert_alpha())

generation = 0

class Player:
    MAX_ROTATION = 25
    ANIMATION_IMAGES = player_images
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.animation_number = 0
        self.img = self.ANIMATION_IMAGES[0]
        self.jumping = False

    def jump(self):
        self.vel = PLAYER_JUMP_VELOCITY
        self.tick_count = 0
        self.height = self.y

    def move(self):
        self.tick_count += 1

        # acceleration
        displacement = self.vel * self.tick_count + 0.5 * 3 * (self.tick_count ** 2)

        # terminal velocity
        if displacement >= PLAYER_TERMINAL_VELOCITY:
            displacement = (displacement/abs(displacement)) * PLAYER_TERMINAL_VELOCITY

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        self.animation_number += 1

        # For animation of bird
        if self.animation_number <= self.ANIMATION_TIME:
            self.img = self.ANIMATION_IMAGES[0]
        elif self.animation_number <= self.ANIMATION_TIME*2:
            self.img = self.ANIMATION_IMAGES[1]
        elif self.animation_number <= self.ANIMATION_TIME*3:
            self.img = self.ANIMATION_IMAGES[2]
        elif self.animation_number <= self.ANIMATION_TIME*4:
            self.img = self.ANIMATION_IMAGES[1]
        elif self.animation_number == self.ANIMATION_TIME*4 + 1:
            self.img = self.ANIMATION_IMAGES[0]
            self.animation_number = 0

        # when bird is nose diving it isn't flapping
        if self.tilt <= -80:
            self.img = self.ANIMATION_IMAGES[1]
            self.animation_number = self.ANIMATION_TIME * 2

        # tilt the bird
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Pipe:
    GAP = 160
    VEL = 7

    def __init__(self, x):
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(pipe_image, False, True)
        self.PIPE_BOTTOM = pipe_image

        self.passed = False

        self.set_height()

    def set_height(self):
        # generate a random number
        self.height = random.randrange(50, 450)
        # offset the top of the pipe from the random number
        self.top = self.height - self.PIPE_TOP.get_height()
        # offset the bottom of the pipe from the random number
        self.bottom = self.height + self.GAP

    def move(self):
        # move the pipe along the screen
        self.x -= self.VEL

    def draw(self, win):
        # draw top
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # draw bottom
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))


    def collide(self, bird):
        # check the collision using pygame masks
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        # overlay the two masks and check if any point are overlapping
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        # if there is any overlap, then there is a collision
        if b_point or t_point:
            return True

        return False

class Ground:
    GROUND_VELOCITY = 5
    WIDTH = ground_image.get_width()
    IMG = ground_image

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1 -= self.GROUND_VELOCITY
        self.x2 -= self.GROUND_VELOCITY
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        # draw two copies of the ground to fill the screen
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, top_left, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=top_left).center)

    surf.blit(rotated_image, new_rect.topleft)

def draw_game(win, birds, pipes, base, score, generation, pipe_ind):
    if generation == 0:
        generation = 1
    win.blit(background_image, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        # draw lines from bird to pipe
        if BIRD_TO_PIPE_LINES:
            try:
                pygame.draw.line(win, (255, 0, 0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255, 0, 0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        # draw bird
        bird.draw(win)

    # score
    score_label = STAT_FONT.render("Score: " + str(score), True, (255, 255, 255))
    win.blit(score_label, (WIDTH - score_label.get_width() - 15, 10))

    # generations
    score_label = STAT_FONT.render("Gens: " + str(generation - 1), True, (255, 255, 255))
    win.blit(score_label, (10, 10))

    # alive
    score_label = STAT_FONT.render("Alive: " + str(len(birds)), True, (255, 255, 255))
    win.blit(score_label, (10, 50))

    pygame.display.update()

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

def genome_evaluation(genomes, config):
    global WINDOW, generation
    win = WINDOW
    generation += 1

    false_positive = 0
    false_negative = 0
    true_positive = 0
    true_negative = 0

    # start by creating lists holding the genome itself, the
    # neural network associated with the genome and the
    # bird object that uses that network to play
    nets = []
    birds = []
    ge = []
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net) #add genome number of nets returned from the library into the nets list
        birds.append(Player(230, 350)) # add genome number of bird instances into the birds lise
        ge.append(genome)

    base = Ground(FLOOR) # create the base of the map
    pipes = [Pipe(700)] #create pipes
    score = 0

    clock = pygame.time.Clock()

    run = True

    # start game loop
    while run and len(birds) > 0:
        clock.tick(30)

        # get events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determine whether to use the first or second
                pipe_ind = 1                                                                 # pipe on the screen for neural network input

        for x, bird in enumerate(birds):  # give each bird a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1
            bird.move()
            # send bird location, top pipe location and bottom pipe location and determine from network whether to jump or not
            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            if output[0] > 0.5:  # tan-h activation function. If over 0.5, bird will jump
                bird.jump()
                bird.jumping = True
            else:
                bird.jumping = False

        base.move()

        pipes_to_remove = []
        add_pipe = False
        # for all pipes in the list
        for pipe in pipes:
            pipe.move()
            # for each bird in the bird list
            for bird in birds:
                # check for collision
                if pipe.collide(bird):
                    ge[birds.index(bird)].fitness -= 1 # reduce fitness
                    nets.pop(birds.index(bird)) # remove the 'dead' birds from the net, genome, and birds list
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))
                    if bird.jumping: # bird jumped and died
                        false_positive += 1
                    elif not bird.jumping: # bird did not jump and died
                        false_negative += 1
                else:
                    if bird.jumping: # bird jumped and lived
                        true_positive += 1
                    elif not bird.jumping: # bird did not jump and lived
                        true_negative += 1

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                pipes_to_remove.append(pipe)

            # this is a stupid warning, so it can be ignored
            # set the indicator to add a pipe if it passes the conditions
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        # if the indicator has been set
        if add_pipe:
            score += 1
            # can add this line to give more reward for passing through a pipe (not required)
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(WIDTH))

        for pipe_to_remove in pipes_to_remove:
            pipes.remove(pipe_to_remove)

        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_game(WINDOW, birds, pipes, base, score, generation, pipe_ind)

        # print the bird's individual confusion matrix when it dies
        # print('                 | Bird Should Jump | Bird Should Not Jump|')
        # print('Bird Jumped      |{:18}|{:21}|'.format(true_positive, false_positive))
        # print('Bird Did Not Jump|{:18}|{:21}|'.format(false_negative, true_negative))

        # score limit check
        if score > SCORE_LIMIT:
            break

    genome_accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)
    genome_precision = true_positive/(true_positive + false_positive)
    genome_recall = true_positive/(true_positive + false_negative)
    genome_f1 = 2 * ((genome_precision * genome_recall)/(genome_precision + genome_recall))

    accuracy_list.append(genome_accuracy)
    precision_list.append(genome_precision)
    recall_list.append(genome_recall)
    f1_list.append(genome_f1)

def run_neat_algorithm(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run for up to 50 generations.
    winner = p.run(genome_evaluation, NUM_OF_GENERATIONS)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner)
    visualize.plot_stats(stats)
    visualize.plot_species(stats)

    figure, axis = plt.subplots(2, 2)
    figure.suptitle('445 Project')

    axis[0, 0].plot(accuracy_list)
    axis[0, 0].set_title('Accuracy')

    axis[0, 1].plot(precision_list)
    axis[0, 1].set_title('Precision')

    axis[1, 0].plot(recall_list)
    axis[1, 0].set_title('Recall')

    axis[1, 1].plot(f1_list)
    axis[1, 1].set_title('F1 Score')

    plt.show()

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run_neat_algorithm(config_path)
