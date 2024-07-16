# Main Code
# Import Required Libraries 
import keyboard
from Functions import loading_model
from Functions import corners
from Functions import prespective_view
from Functions import grid
from Functions import detection
from Functions import fenannotation
from Functions import FEN
from Functions import bestmove
from Functions import take_photo
from Functions import chess_move_to_indices
from Functions import position
from Functions import fifo
from Functions import check_end_game
from Functions import ziel_func


from Functions import ads_aufbau

from stockfish import Stockfish
import chess.svg
import os
import cv2
import pyads
from ctypes import sizeof
########################################
# Construction of the ads communication (between PLC and PC)
ads_net_id='143.93.155.145.1.1'
plc=pyads.Connection(ads_net_id,pyads.PORT_TC3PLC1)
plc.open()
print(plc.read_state()) ##(5,0)= connected
#######################################
# start stockfish Engine
stockfish=Stockfish(r"stockfish\stockfish-windows-x86-64-avx2.exe")
stockfish.set_skill_level(15) #set the engine level
image='corners.pgm' # image name for coner detection code
di = {0: 'b', 1: 'k', 2: 'n',   #b=SchwarzLaeufer, k=SchwarzKoenig, n=SchwarzSpringer
      3: 'p', 4: 'q', 5: 'r',   #p=SchwarzBauer,   q=SchwarzDame,   r=SchwarzTurm
      6: 'B', 7: 'K', 8: 'N',   #B=WeissLaeufer,   K=WeissKoenig,   N=WeissSpringer
      9: 'P', 10: 'Q', 11: 'R'} #P=WeissBauer,     Q=WeissDame,     R=WeissTurm

#load corner model #amisadik51
C_model=loading_model("jT1lVWPhIoaDuajgutxG", "schachbretteckenerkennung")
D_model=loading_model("jT1lVWPhIoaDuajgutxG", "schachfigurenerkennung")

#load detection model #amisadik52
#D_model=loading_model("qwery01EKaECu2Ah9hjy", "schachfiguren-erkennung-nwjy4")


if  True:
    while True:
        rect=corners(image, C_model)

        
        for i in range(10):
            plc.write_by_name(data_name='gvl50_einpy.StartHandshakeEingang',value=True,plc_datatype=pyads.PLCTYPE_BOOL) ##Handshake starten
            plc.write_by_name(data_name='gvl50_einpy.Running',value=True,plc_datatype=pyads.PLCTYPE_BOOL) ##Programm läuft
            take_photo('figures.pgm')
            image='figures.pgm'  #will be replaced with camera take image function
            img = prespective_view(rect, image)
            [ptsT, ptsL, ptsR, ptsB]= grid(img)
            img.save("Perspektivischen_Transformationsmatrix.jpg") # Speichern perspektivisches Bild
            dres = D_model.predict("Perspektivischen_Transformationsmatrix.jpg", confidence=50, overlap=50).json()
            #D_model.predict("Perspektivischen_Transformationsmatrix.jpg", confidence=40, overlap=50).save('Figurenerkennung.jpg')
            detections, classes = detection(dres)
            FEN_annotation = fenannotation(ptsT, ptsL, ptsR, ptsB)
            fen_position = FEN(detections, FEN_annotation, classes, di)
            #print(fen_position)
            best= bestmove(fen_position, stockfish)
            print(f'best_move?: {best}')
            isitfree = fifo(fen_position, best, stockfish)
            #ziel = ziel_func(fen_position, best)
            print(f'end_position_possessed?: {isitfree}')
            whoknow = check_end_game(fen_position)
            print(f'end_game?:{whoknow}')

            # Example usage
            source, dest = chess_move_to_indices(best)
            #print("Source indices:", source)
            #print("Destination indices:", dest)

            Ibox = [133,-520,150]
            d= [35,35]
            sourcep, destp = position(source, dest, Ibox, d)
            #print(sourcep,destp)
            print(f'init_pos:{sourcep}, end_pos:{destp}')
            
            if isitfree[0] is None:
                plc.write_by_name(data_name='gvl260_AusRob.Move_type',value=1,plc_datatype=pyads.PLCTYPE_BYTE) ##Zugart übergeben
            else:
                plc.write_by_name(data_name='gvl260_AusRob.Move_type',value=2,plc_datatype=pyads.PLCTYPE_BYTE) ##Zugart übergeben
                
            
            
            

            Rdy=plc.read_by_name(data_name='gvl250_auspy.RDYDatenEmpfang',plc_datatype=pyads.PLCTYPE_BOOL)
            if Rdy == True:
                
                plc.write_by_name(data_name='gvl50_einpy.PoseGeg[1]',value=sourcep[0],plc_datatype=pyads.PLCTYPE_REAL) ##x1_koordinate Übergeben
                plc.write_by_name(data_name='gvl50_einpy.PoseGeg[2]',value=sourcep[1],plc_datatype=pyads.PLCTYPE_REAL) ##y1_koordinate Übergeben
                plc.write_by_name(data_name='gvl50_einpy.PoseGeg[3]',value=destp[0],plc_datatype=pyads.PLCTYPE_REAL) ##x2_koordinate Übergeben
                plc.write_by_name(data_name='gvl50_einpy.PoseGeg[4]',value=destp[1],plc_datatype=pyads.PLCTYPE_REAL) ##y1_koordinate Übergeben
                plc.write_by_name(data_name='gvl50_einpy.DatenGesendet',value=True,plc_datatype=pyads.PLCTYPE_BOOL) ##Daten gesendet

            b=bool(True)
            while b:
                Start=plc.read_by_name(data_name='gvl250_auspy.PythonStart',plc_datatype=pyads.PLCTYPE_BOOL)
                if Start == True: 
                 #if  keyboard.is_pressed('space'):
                    b=bool(False)  
        
            
