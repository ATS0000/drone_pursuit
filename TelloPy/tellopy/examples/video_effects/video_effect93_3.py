#r -> celect region of tracked object -> Enter


import sys
import traceback
import tellopy
import av
import cv2#.cv2 as cv2  # for avoidance of pylint error
import numpy
import time
from time import sleep
import math as m


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def tracking_near(drone,d,dx,dy,LB):
    """
    if (d - LB) > 15:
        drone.set_pitch(5)   #drone.pitch = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    elif (d - LB) < -15:
        drone.set_pitch(-5)   #drone.pitch = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)
    """
    gain_x = 0.2
    gain_y = 0.1

    if dx > 0:  #80
	print('right')
        drone.right(gain_x*dx)
	#drone.roll = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    if dx < 0:#-80
	print('left')
        drone.left(- gain_x*dx)    #drone.roll = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)
    if dy > 0:
	print('down')
        drone.down(gain_y*dy)    #drone.thr = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    if dy < 0:
	print('up')
        drone.up(- gain_y*dy)    #drone.thr = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)


def tracking_midle(drone,d,dx,dy,LB):
    """
    if (d - LB) > 15:
        drone.set_pitch(5)   #drone.pitch = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    elif (d - LB) < -15:
        drone.set_pitch(-5)   #drone.pitch = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)
    """
    gain_x = 0.00025
    gain_y = 0.0001

    if dx > 0:  #80
	print('right')
        drone.right(gain_x*dx*dx + 5)
	#drone.roll = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    if dx < 0:#-80
	print('left')
        drone.left(gain_x*dx*dx + 5)    #drone.roll = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)
    if dy > 0:
	print('down')
        drone.down(gain_y*dy*dy + 5)    #drone.thr = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    if dy < 0:
	print('up')
        drone.up(gain_y*dy*dy + 5)    #drone.thr = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)


def tracking_far(drone,d,dx,dy,LB):
    """
    if (d - LB) > 15:
        drone.set_pitch(5)   #drone.pitch = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    elif (d - LB) < -15:
        drone.set_pitch(-5)   #drone.pitch = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)
    """
    gain_x = 0.1
    gain_y = 0.05

    #sleep(1)

    if dx > 0:  #80
	print('right')
        drone.right(gain_x*dx + 15)
	#drone.roll = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    if dx < 0:#-80
	print('left')
        drone.left(- gain_x*dx + 15)    #drone.roll = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)
    if dy > 0:
	print('down')
        drone.down(gain_y*dy + 15)    #drone.thr = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    if dy < 0:
	print('up')
        drone.up(- gain_y*dy + 15)    #drone.thr = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)


def main():

    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

    x0 = 200
    y0 = 200
    w0 = 224
    h0 = 224
    track_window=(x0,y0,w0,h0)
    # Reference Distance
    L0 = 100
    S0 = 50176 #224x224 #take#here.

    # Base Distance
    LB = 100
    # Define an initial bounding box
    bbox = (x0, y0, w0, h0)   #(287, 23, 86, 320)
    #CX=int(bbox[0]+0.5*bbox[2]+3) #adding
    #CY=int(bbox[1]+0.5*bbox[3]+3) #adding





    drone = tellopy.Tello()

    try:
        drone.connect()
        drone.wait_for_connection(60.0)

        retry = 3
        container = None
        while container is None and 0 < retry:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as ave:
                print(ave)
                print('retry...')

        drone.takeoff()

        # skip first 300 frames
        frame_skip = 300
        while True:
#------------------------------------------for start

            for frame in container.decode(video=0):

		speed = 100
		
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
		
		start_time = time.time()

		image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)

		# Start timer
                timer = cv2.getTickCount()

                #cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                #cv2.waitKey(1)

		# Update tracker
                ok, bbox = tracker.update(image)

		# Calculate Frames per second (FPS)
                fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);


# Draw bounding box
                if ok:
		    #print('Tracking ok')
                    (x,y,w,h) = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
                    CX=int(bbox[0]+0.5*bbox[2]) #Center of X
                    CY=int(bbox[1]+0.5*bbox[3])
                    S0=bbox[2]*bbox[3]
                    print("CX,CY,S0,x,y=",CX,CY,S0,x,y)
                    # Tracking success
                    p1 = (x, y)
                    p2 = (x + w, y + h)
                    cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)
		    p10 = (x0, y0)
                    p20 = (x0 + w0, y0 + h0)
                    cv2.rectangle(image, p10, p20, (0,255,0), 2, 1)

                    d = round(L0 * m.sqrt(S0 / (w * h)))
                    dx = x + w/2 - CX0 
                    dy = y + h/2 - CY0
                    print(d,dx,dy)

		    if abs(x + w/2 - 500) <= 100:
		        tracking_near(drone,d,dx,dy,LB)

		    elif 100 < abs(x + w/2 - 500) <= 250:
		        tracking_midle(drone,d,dx,dy,LB)

		    elif 250 < abs(x + w/2 - 500):
		        tracking_far(drone,d,dx,dy,LB)


		    
                else:
                # Tracking failure
		    #print('Tracking failure')
                    cv2.putText(image, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

                cv2.imshow('Original', image)

		key = cv2.waitKey(1)&0xff
		if key == ord('q'):
		    print('Q!')
		    break

		if key == ord('r'):
		    roi_time = time.time()
                    bbox = cv2.selectROI(image, False)
                    print(bbox)
                    (x0,y0,w0,h0) = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))

                    CX0=int(x0+0.5*w0) #Center of X
                    CY0=int(y0+0.5*h0)
		    
                    # Initialize tracker with first frame and bounding box
                    ok = tracker.init(image, bbox)

		    '''
		    if frame.time_base < 1.0/60:
                        time_base = 1.0/60
                    else:
                        time_base = frame.time_base
                    frame_skip2 = int((time.time() - roi_time)/time_base)

		    if 0 < frame_skip2:
                        frame_skip2 = frame_skip2 - 1
                        continue
		    '''

		if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)

		    



#-------------------------------------------------for end
            break
	print('stop fly')
                    

    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
	drone.land()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
