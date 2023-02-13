# Quarto-Final-Project

This repository is allocated for the final project of Computational Intelligence course. 

## About the Project
The project is about a game named Quarto which is a two-player game and has a board and 16 pieces. The pieces have four characteristics (High/Not High, Colored/Not Colored, Square/Not Square, Solid/ Not Solid). Players choose a piece for their opponent to play and the apponent chooses the best possible position for it on the board. The player who places 4 pieces with the same characteristics in a row, column or diagonal to form a Quarto, is the winner.

## Approach: Reinfocement Learning<br>

We run the code developed in an iterative manner to learn the weights. Then, we use the weights achieved to compete with a random opponent.

## Results

The best weights I achieved for the pieces and places are reported below:

### Pieces
>{</br>
>0: -198.94833926724817,</br> 1: -281.1166312781826,</br> 2: -152.34285903736216,</br> 3: -76.6750826495723,</br> 4: -80.10840879570897,</br> 5: -75.52529948493937,</br> 6: -122.50325973280344,</br> 7: -160.60942116965253,</br> 8: -82.80408320918029,</br> 9: -138.7928694906595,</br> 10: -144.47109628778782,</br> 11: -116.15021526230467,</br> 12: -102.31958181624456,</br> 13: -229.51831557660017,</br> 14: -178.09552234779338,</br> 15: -264.03911617024403}

### Places
>{</br>
>(0, 0): -155.52884626044155,</br> (0, 1): -175.4187144036783,</br> (0, 2): -218.89089183630927,</br> (0, 3): -186.0244466067697,</br> (1, 0): -284.13366245401204,</br> (1, 1): -228.8937470788861,</br> (1, 2): -220.1238998925363,</br> (1, 3): -287.80559138470676,</br> (2, 0): -342.59855306705896,</br> (2, 1): -252.27851652829716,</br> (2, 2): -175.24755254355844,</br> (2, 3): -202.5267674807503,</br> (3, 0): -239.47477242244548,</br> (3, 1): -254.25774859497977,</br> (3, 2): -307.1145869403,</br> (3, 3): -222.13636169056804}


### Figures

![result1!](/figures/Figure_4.jpeg "last")
