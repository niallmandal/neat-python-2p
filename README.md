## About `neat-python` ##

NEAT (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural 
networks. This project is a Python implementation of NEAT.  It was forked from the excellent project by @MattKallada, 
and is in the process of being updated to provide more features and a (hopefully) simpler and documented API.

For further information regarding general concepts and theory, please see [Selected Publications](http://www.cs.ucf.edu/~kstanley/#publications) on Stanley's website.

`neat-python` is licensed under the [3-clause BSD license](https://opensource.org/licenses/BSD-3-Clause).

## What this project is ##
Im simple terms, a file built on top of `neat-python` to run two NEAT agents simultaneously. This is exemplified in a 2 player veriosn of PONG.
This project is a compilation of two things:
1. An edit of a [file](https://github.com/CodeReclaimers/neat-python/blob/master/neat/population.py) from `neat-python` based on the actual creation of each NEAT population
2. An example of an implementation of this new file using **PONG** on the Atari 2600 with emulation via OpenAI Retro

## twoplayer ##
This is basically just a folder containg the custom code built on top of `neat-python`. It alters `population.py` in such a way that two independent populations are run per instance.

## PONG-2p ##
This is the file that actually runs two populations against eachother in a Video Game environment. Input data are metrics derived from the game based on the current score, and rectangular coordinates (eventually converted to polar coordinates) of sprites in the game.



***NOTE:** in order to run this code, you must already have OpenAI Retro installed on your computer. The _Pong-Atari2600_ is a folder that goes in the `/data` folder of retro

Ex: `C:\Users\john-doe\gym-retro\retro\data\stable\Pong-Atari2600`
