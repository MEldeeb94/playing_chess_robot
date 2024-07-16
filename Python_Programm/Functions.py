# Functions 

from matplotlib.pyplot import figure
import matplotlib.image as image
from matplotlib import pyplot as plt


import pandas as pd #
import numpy as np
from numpy import asarray
from PIL import Image

import cv2


from shapely.geometry import Polygon

from IPython.display import display

from roboflow import Roboflow


import chess
import chess.svg

import os
import shutil
import subprocess
import pyads
from ctypes import sizeof

def ads_aufbau():
    ads_net_id='143.93.155.145.1.1' # Für Rechner 1 ADS = 143.93.155.128.1.1
    plc=pyads.Connection(ads_net_id,pyads.PORT_TC3PLC1)

    plc.open()

    print(plc.read_state()) ##(5,0)= verbunden


def chess_move_to_indices(move):

    # Mapping from chess notation to matrix indices
    column_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
    row_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7}

    # Extracting source and destination indices
    source = [column_map[move[0]], row_map[move[1]]]
    dest = [column_map[move[2]], row_map[move[3]]]
 
    return source, dest

def position(source, dest, Ibox, d):

    Iboxx= Ibox[0]
    Iboxy= Ibox[1]
    Iboxz = Ibox[2]
    dx = d[0]
    dy = d[1]

    x = [Iboxx-0.5*dx, Iboxx-1.5*dx, Iboxx-2.5*dx, Iboxx-3.5*dx, Iboxx-4.5*dx, Iboxx-5.5*dx, Iboxx-6.5*dx, Iboxx-7.5*dx]
    y = [Iboxy+0.5*dy, Iboxy+1.5*dy, Iboxy+2.5*dy, Iboxy+3.5*dy, Iboxy+4.5*dy, Iboxy+5.5*dy, Iboxy+6.5*dy, Iboxy+7.5*dy]
    
    sourcep = [x[source[1]], y[source[0]], Iboxz]
    destp = [x[dest[1]], y[dest[0]], Iboxz]

    return sourcep, destp


def take_photo(name):
    #command = "rc_visard_record.bat -n 1  -right false -confidence false -disparity false -error false 07009986"
    command = "rc_visard_record.bat -n 1  -right false -confidence false -disparity false -error false 00012ea68dde"
    subprocess.run(command, shell=True)


    # Define the source and destination folders
    destination_folder = r"C:\Users\maadmin\Desktop\Schachprogramm\Python_Programm"

    # Get a list of all items in the source folder
    items = os.listdir(destination_folder)
    #image_folder = [item for item in items if '07009986' in item]
    image_folder = [item for item in items if '00012ea68dde' in item]
    source_folder = f"{destination_folder}\\{image_folder[0]}"
    images = os.listdir(source_folder)
    image = [image for image in images if '.pgm' in image]

    image_path = f"{source_folder}\\{image[0]}"

    d_image = f"{destination_folder}\\{name}"
    shutil.copyfile(image_path,d_image)
    shutil.rmtree(source_folder)



def loading_model(api_code, project):
    rf = Roboflow(api_key=api_code)
    project = rf.workspace().project(project)
    model = project.version(1).model #C_model "model für Erkennung der Ecken"
    return model



def corners(image, C_model):
    take_photo('corners.pgm')
    rect=[]
    while len(rect) != 4:
        results=C_model.predict(image, confidence=0.02, overlap=30)
        a= results[0]
        b= results[1]
        c= results[2]
        d= results[3]

        offset = 0
        corners = np.array([[a["x"]-offset,a["y"]-offset], 
                            [b["x"]+offset,b["y"]-offset], 
                            [c["x"]+offset,c["y"]+offset], 
                            [d["x"]-offset,d["y"]+offset]])
        #print(corners)
        s = corners.sum(axis = 1)
        rect = np.zeros((4, 2), dtype = "float32")
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]

        diff = np.diff(corners, axis = 1) #y-x
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]
        # take new image
    return rect

def prespective_view(rect, image):
    
    cc = np.empty((4,2))
    # X Axis
    cc[0,0]=rect[0,0]-5       #TL
    cc[1,0]=rect[1,0]+5       #TR
    cc[2,0]=rect[2,0]+8       #BR
    cc[3,0]=rect[3,0]-20      #BL

    # Y Axis
    cc[0,1]=rect[0,1]-67      #TL
    cc[1,1]=rect[1,1]-67      #TR
    cc[2,1]=rect[2,1]-2       #BR
    cc[3,1]=rect[3,1]-2       #BL


    (tl, tr, br, bl) = cc # bl=bottom-left, tl=top-left, tr=top-right, br=bottom-right

    # Berechnen der Breite des neuen Bildes 
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))


    # Berechnen der Höhe des neuen Bildes
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # construct set of destination points to obtain a "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    img = Image.open(image)
    img = asarray(img)
    # Berechnung der perspektivischen Transformationsmatrix und anschließend beschneiden das Bild (Croping the image)
    #print(cc)
    #print(cc.shape)
    cc = cc.astype(np.float32)
    M = cv2.getPerspectiveTransform(cc, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    warped = np.dstack((warped, warped, warped))

    img = Image.fromarray(warped, "RGB")

    return img

def grid(img):
    ss=img.size
    figure(figsize=(10, 10), dpi=80)

    implot = plt.imshow(img)

    TL=[22,85]
    BL=[20,ss[1]-5]
    TR=[ss[0]-20,85]
    BR=[ss[0]-10,ss[1]-5]

    def interpolate(xy0, xy1):
        x0,y0 = xy0
        x1,y1 = xy1
        dx = (x1-x0) / 8
        dy = (y1-y0) / 8
        pts = [(x0+i*dx,y0+i*dy) for i in range(9)]
        return pts

    ptsT = interpolate( TL, TR )
    ptsL = interpolate( TL, BL )
    ptsR = interpolate( TR, BR )
    ptsB = interpolate( BL, BR )
        
    for a,b in zip(ptsL, ptsR):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
    for a,b in zip(ptsT, ptsB):
        plt.plot( [a[0], b[0]], [a[1], b[1]], 'ro', linestyle="--" )
        
    plt.axis('off')

    plt.savefig('Perspektivischen_Transformationsmatrix_Mit_Grid.jpg')

    return ptsT, ptsL, ptsR, ptsB

def detection(dres):
    # Indizes der Vorhersagen erhalten
    d=dres["predictions"]
    detections= []
    classes= []
    for e in d:
        detections.append([e["x"]-e["width"],e["y"]-e["height"],e["x"]+e["width"],e["y"]+e["height"]])
        c=e["class"]
        if c=='SchwarzLaeufer':
            cc=0
        if c=='SchwarzKoenig':
            cc=1
        if c=='SchwarzSpringer':
            cc=2
        if c=='SchwarzBauer':
            cc=3
        if c=='SchwarzDame':
            cc=4
        if c=='SchwarzTurm':
            cc=5

        if c=='WeissLaeufer':
            cc=6
        if c=='WeissKoenig':
            cc=7
        if c=='WeissSpringer':
            cc=8
        if c=='WeissBauer':
            cc=9
        if c=='WeissDame':
            cc=10
        if c=='WeissTurm':
            cc=11
                

        classes.append(cc)
    detections=np.array(detections)
    detections = detections.reshape(len(detections),4)

    detections[:,1]=detections[:,1]-50
    detections[:,3]=detections[:,3]-50
    return detections, classes
# Brechnung iou zwischen 2 polygons

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

def connect_square_to_detection(detections, square, classes):
    
    di = {0: 'b', 1: 'k', 2: 'n',   #b=SchwarzLaeufer, k=SchwarzKoenig, n=SchwarzSpringer
          3: 'p', 4: 'q', 5: 'r',   #p=SchwarzBauer,   q=SchwarzDame,   r=SchwarzTurm
          6: 'B', 7: 'K', 8: 'N',   #B=WeissLaeufer,   K=WeissKoenig,   N=WeissSpringer
          9: 'P', 10: 'Q', 11: 'R'} #P=WeissBauer,     Q=WeissDame,     R=WeissTurm

    list_of_iou=[]
    
    for i in detections:

        box_x1 = i[0]
        box_y1 = i[1]

        box_x2 = i[2]
        box_y2 = i[1]

        box_x3 = i[2]
        box_y3 = i[3]

        box_x4 = i[0]
        box_y4 = i[3]
        
        #Schneiden Höhe Figuren        
        if box_y4 - box_y1 > 60:
            box_complete = np.array([[box_x1,box_y4 -60], [box_x2, box_y3 -60], [box_x3, box_y3], [box_x4, box_y4]])
        else:
            box_complete = np.array([[box_x1,box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])

        #Schneiden Höhe Figuren 
        '''       
        if box_complete[1,0] - box_complete[0,0] > 80:
            box_complete[0,0] = box_complete[1,0]-70
            box_complete[3,0] = box_complete[2,0]-70
        '''
        #Schneiden Höhe Figuren        
        #if box_x1 - box_x2 > 30:
        #    box_complete = np.array([[box_x1 ,box_y4], [box_x2, box_y2], [box_x3+70, box_y3], [box_x4+70 , box_y4]])
        #else:
        #    box_complete = np.array([[box_x1,box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])
          
        #bis

        list_of_iou.append(calculate_iou(box_complete, square))

    

    #piece = classes[num]
    #print(piece)
    if max(list_of_iou) > 0.22:
        num = list_of_iou.index(max(list_of_iou))
        piece = classes[num]
        return di[piece]
    
    else:
        piece = "empty"
        return piece
    
def fenannotation(ptsT, ptsL, ptsR, ptsB):
    #Berechnung des Schachbrettes

    xA = ptsT[0][0]
    xB = ptsT[1][0]
    xC = ptsT[2][0]
    xD = ptsT[3][0]
    xE = ptsT[4][0]
    xF = ptsT[5][0]
    xG = ptsT[6][0]
    xH = ptsT[7][0]
    xI = ptsT[8][0]

    y9 = ptsL[0][1]
    y8 = ptsL[1][1] 
    y7 = ptsL[2][1] 
    y6 = ptsL[3][1]  
    y5 = ptsL[4][1]  
    y4 = ptsL[5][1] 
    y3 = ptsL[6][1]  
    y2 = ptsL[7][1] 
    y1 = ptsL[8][1] 

        #Berechnung aller Felder

    a8 = np.array([[xA,y9], [xB, y9], [xB, y8], [xA, y8]])
    a7 = np.array([[xA,y8], [xB, y8], [xB, y7], [xA, y7]])
    a6 = np.array([[xA,y7], [xB, y7], [xB, y6], [xA, y6]])
    a5 = np.array([[xA,y6], [xB, y6], [xB, y5], [xA, y5]])
    a4 = np.array([[xA,y5], [xB, y5], [xB, y4], [xA, y4]])
    a3 = np.array([[xA,y4], [xB, y4], [xB, y3], [xA, y3]])
    a2 = np.array([[xA,y3], [xB, y3], [xB, y2], [xA, y2]])
    a1 = np.array([[xA,y2], [xB, y2], [xB, y1], [xA, y1]])

    b8 = np.array([[xB,y9], [xC, y9], [xC, y8], [xB, y8]])
    b7 = np.array([[xB,y8], [xC, y8], [xC, y7], [xB, y7]])
    b6 = np.array([[xB,y7], [xC, y7], [xC, y6], [xB, y6]])
    b5 = np.array([[xB,y6], [xC, y6], [xC, y5], [xB, y5]])
    b4 = np.array([[xB,y5], [xC, y5], [xC, y4], [xB, y4]])
    b3 = np.array([[xB,y4], [xC, y4], [xC, y3], [xB, y3]])
    b2 = np.array([[xB,y3], [xC, y3], [xC, y2], [xB, y2]])
    b1 = np.array([[xB,y2], [xC, y2], [xC, y1], [xB, y1]])

    c8 = np.array([[xC,y9], [xD, y9], [xD, y8], [xC, y8]])
    c7 = np.array([[xC,y8], [xD, y8], [xD, y7], [xC, y7]])
    c6 = np.array([[xC,y7], [xD, y7], [xD, y6], [xC, y6]])
    c5 = np.array([[xC,y6], [xD, y6], [xD, y5], [xC, y5]])
    c4 = np.array([[xC,y5], [xD, y5], [xD, y4], [xC, y4]])
    c3 = np.array([[xC,y4], [xD, y4], [xD, y3], [xC, y3]])
    c2 = np.array([[xC,y3], [xD, y3], [xD, y2], [xC, y2]])
    c1 = np.array([[xC,y2], [xD, y2], [xD, y1], [xC, y1]])

    d8 = np.array([[xD,y9], [xE, y9], [xE, y8], [xD, y8]])
    d7 = np.array([[xD,y8], [xE, y8], [xE, y7], [xD, y7]])
    d6 = np.array([[xD,y7], [xE, y7], [xE, y6], [xD, y6]])
    d5 = np.array([[xD,y6], [xE, y6], [xE, y5], [xD, y5]])
    d4 = np.array([[xD,y5], [xE, y5], [xE, y4], [xD, y4]])
    d3 = np.array([[xD,y4], [xE, y4], [xE, y3], [xD, y3]])
    d2 = np.array([[xD,y3], [xE, y3], [xE, y2], [xD, y2]])
    d1 = np.array([[xD,y2], [xE, y2], [xE, y1], [xD, y1]])

    e8 = np.array([[xE,y9], [xF, y9], [xF, y8], [xE, y8]])
    e7 = np.array([[xE,y8], [xF, y8], [xF, y7], [xE, y7]])
    e6 = np.array([[xE,y7], [xF, y7], [xF, y6], [xE, y6]])
    e5 = np.array([[xE,y6], [xF, y6], [xF, y5], [xE, y5]])
    e4 = np.array([[xE,y5], [xF, y5], [xF, y4], [xE, y4]])
    e3 = np.array([[xE,y4], [xF, y4], [xF, y3], [xE, y3]])
    e2 = np.array([[xE,y3], [xF, y3], [xF, y2], [xE, y2]])
    e1 = np.array([[xE,y2], [xF, y2], [xF, y1], [xE, y1]])

    f8 = np.array([[xF,y9], [xG, y9], [xG, y8], [xF, y8]])
    f7 = np.array([[xF,y8], [xG, y8], [xG, y7], [xF, y7]])
    f6 = np.array([[xF,y7], [xG, y7], [xG, y6], [xF, y6]])
    f5 = np.array([[xF,y6], [xG, y6], [xG, y5], [xF, y5]])
    f4 = np.array([[xF,y5], [xG, y5], [xG, y4], [xF, y4]])
    f3 = np.array([[xF,y4], [xG, y4], [xG, y3], [xF, y3]])
    f2 = np.array([[xF,y3], [xG, y3], [xG, y2], [xF, y2]])
    f1 = np.array([[xF,y2], [xG, y2], [xG, y1], [xF, y1]])

    g8 = np.array([[xG,y9], [xH, y9], [xH, y8], [xG, y8]])
    g7 = np.array([[xG,y8], [xH, y8], [xH, y7], [xG, y7]])
    g6 = np.array([[xG,y7], [xH, y7], [xH, y6], [xG, y6]])
    g5 = np.array([[xG,y6], [xH, y6], [xH, y5], [xG, y5]])
    g4 = np.array([[xG,y5], [xH, y5], [xH, y4], [xG, y4]])
    g3 = np.array([[xG,y4], [xH, y4], [xH, y3], [xG, y3]])
    g2 = np.array([[xG,y3], [xH, y3], [xH, y2], [xG, y2]])
    g1 = np.array([[xG,y2], [xH, y2], [xH, y1], [xG, y1]])

    h8 = np.array([[xH,y9], [xI, y9], [xI, y8], [xH, y8]])
    h7 = np.array([[xH,y8], [xI, y8], [xI, y7], [xH, y7]])
    h6 = np.array([[xH,y7], [xI, y7], [xI, y6], [xH, y6]])
    h5 = np.array([[xH,y6], [xI, y6], [xI, y5], [xH, y5]])
    h4 = np.array([[xH,y5], [xI, y5], [xI, y4], [xH, y4]])
    h3 = np.array([[xH,y4], [xI, y4], [xI, y3], [xH, y3]])
    h2 = np.array([[xH,y3], [xI, y3], [xI, y2], [xH, y2]])
    h1 = np.array([[xH,y2], [xI, y2], [xI, y1], [xH, y1]])

    FEN_annotation = [[a1, a2, a3, a4, a5, a6, a7, a8],
                    [b1, b2, b3, b4, b5, b6, b7, b8],
                    [c1, c2, c3, c4, c5, c6, c7, c8],
                    [d1, d2, d3, d4, d5, d6, d7, d8],
                    [e1, e2, e3, e4, e5, e6, e7, e8],
                    [f1, f2, f3, f4, f5, f6, f7, f8],
                    [g1, g2, g3, g4, g5, g6, g7, g8],
                    [h1, h2, h3, h4, h5, h6, h7, h8]]
    
    return FEN_annotation

def FEN(detections, FEN_annotation, classes, di):
    board_FEN = []
    corrected_FEN = []
    complete_board_FEN = []
    line_to_FEN = []
    z=[]
    a=0
    for i in detections:
        list_of_iou=[]
        box_x1 = i[0]
        box_y1 = i[1]

        box_x2 = i[2]
        box_y2 = i[1]

        box_x3 = i[2]
        box_y3 = i[3]

        box_x4 = i[0]
        box_y4 = i[3]
        
        #Schneiden Höhe Figuren  

        if box_y4 - box_y1 > 30:
            box_complete = np.array([[box_x1,box_y4 -30], [box_x2, box_y3 -30], [box_x3, box_y3], [box_x4, box_y4]])
        else:
            box_complete = np.array([[box_x1,box_y1], [box_x2, box_y2], [box_x3, box_y3], [box_x4, box_y4]])
        for line in FEN_annotation:
            
            for square in line:
                list_of_iou.append(calculate_iou(box_complete, square))
                #piece_on_square = connect_square_to_detection(detections, square, classes)    
                #if max(list_of_iou) > 0.5:
                num = list_of_iou.index(max(list_of_iou))
                piece = classes[a]
                piece=di[piece]
        z.append([num,piece])
        a=a+1
    #print(z)
    corrected_FEN= [1 for _ in range(64)]

    for i in z:
        corrected_FEN[i[0]]= (i[1])

    corrected_FEN=np.array(corrected_FEN)
    corrected_FEN=np.reshape(corrected_FEN,[8,8])
    print(corrected_FEN)
    '''
    line_to_FEN.append(piece)
    corrected_FEN = [i.replace('empty', '1') for i in line_to_FEN]
    print(corrected_FEN)
    '''
    corrected_FEN.tolist()
    #board_FEN.append(corrected_FEN)

    complete_board_FEN = [''.join(line) for line in corrected_FEN] 

    to_FEN = '/'.join(complete_board_FEN)

    fen_position = to_FEN.replace("11111111", "8").replace("1111111", "7").replace("111111", "6").replace("11111", "5").replace("1111", "4").replace("111", "3").replace("11", "2").replace("1", "1")

    fen_position = fen_position + " w - - 0 1"
    print("https://lichess.org/analysis/"+fen_position)


    return fen_position

def bestmove(fen, stockfish):
    stockfish.set_fen_position(fen)
    move= stockfish.get_best_move()

    return move

def fifo (fen, chess_move, stockfish):
    string = chess_move
    end_position = string[2:]  # This will get the rest of the string ''
    #print(end_position)  # Output: 
    possession = stockfish.get_what_is_on_square(end_position)
    
    return possession, end_position



def ziel_func (fen, chess_move):
    string = chess_move
    end_position = string[2:]
    return end_position

def fen_to_board(fen):
    board = chess.Board(fen)
    return board

def check_end_game(fen):
    chess_board = fen_to_board(fen)
    #display(chess_board)
    maybe = chess_board.is_checkmate()
    return maybe
