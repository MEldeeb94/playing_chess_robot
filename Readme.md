# Chess Playing Robot (using KUKA Robot)
This project entails creating a chess-playing robot using a KUKA robot arm, equipped with machine vision to identify the chessboard and pieces. The system integrates the Stockfish engine for board analysis and move decision-making. This setup allows the robot to autonomously play chess with high accuracy and strategic depth.
## Hardware Requirements
1. KUKA Robot controller (KR C5 micro)
2. rc_visard 3D Stereo Sensor
3. PC for processing Camera data and Handle output to PLC
4. PLC for communication betwee PC and KUKA (KR C5 micro)
## Software Elements
1. ADS communication for communication between PLC and PC
2. `took_photo` function for taking new image

   <img src="Images/1-%20image.png" alt="alt text" width="300"/>

3. Machine Vision model for detecting board corners (corner points are updated each 10 games)

   <img src="Images/2-%20Corners.jpg" alt="alt text" width="300"/>

4. Image processing for clipping the took image 

   <img src="Images/3-%20clipped_image.jpg" alt="alt text" width="300"/>

5. Machine Vision model for detecting the chess figures 

   <img src="Images/4-%20Figure_detection.jpg" alt="alt text" width="300"/>

6. Mapping all chess figures to each board square then generating FEN code

   <img src="Images/5-%20chessboard_transformed_with_grid_1.jpg" alt="alt text" width="300"/>

7. Handling FEN to Stockfish Engine
8. Interpreting Stockfish output to robot moving orders and sending it to PLC using ADS

6. Handling FEN to stockfish Engine
7. interpreting stockfish output to robot moving orders and sending it to PLC using ADS

 ![alt text](vid_3_chess.mp4)
