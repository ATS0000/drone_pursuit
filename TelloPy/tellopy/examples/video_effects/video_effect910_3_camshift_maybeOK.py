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


#face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')


def tracking(drone,d,dx,dy,L0):


    gain_bf = 0.0005
    
    if d > 50:
	print('back')
        drone.set_pitch(- gain_bf*d)   #drone.pitch = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    elif d < -50:
	print('forward')
        drone.set_pitch(- gain_bf*d)   #drone.pitch = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)
    
    #speed = 100

    gain_x = 0.0003
    gain_y = 0.00005


    gain_x_1 = 0.15
    gain_y_1 = 0.1


    if 5 < dx <= 150:  #80
	print('right')
        drone.right(gain_x_1*dx)
	#drone.roll = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    if -150 <= dx < -5:#-80
	print('left')
        drone.left(- gain_x*dx)    #drone.roll = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)


    if dx > 150:  #80
	print('right')
        drone.right(gain_x*dx*dx)
	#drone.roll = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    if dx < -150:#-80
	print('left')
        drone.left(gain_x*dx*dx)    #drone.roll = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)
    if dy > 5:
	print('down')
        drone.down(gain_y*dy*dy)    #drone.thr = drone.STICK_HOVER + drone.STICK_L
        #sleep(1)
    elif dy < 5:
	print('up')
        drone.up(gain_y*dy*dy)    #drone.thr = drone.STICK_HOVER - drone.STICK_L
        #sleep(1)



def main():

    '''
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
    '''

	

    #x = 200
    #y = 200
    #w = 224
    #h = 224
    #track_window=(x,y,w,h)
    # Reference Distance
    L0 = 100
    #S0 = 50176 #224x224 #take#here.

    # Base Distance
    #LB = 100
    # Define an initial bounding box
    #bbox = (x0, y0, w0, h0)   #(287, 23, 86, 320)
    #CX=int(bbox[0]+0.5*bbox[2]+3) #adding
    #CY=int(bbox[1]+0.5*bbox[3]+3) #adding

    cap = cv2.VideoCapture(0)

    ok = False


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

		#speed = 100
		
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue

		#ret,image = cap.read()

		start_time = time.time()

		image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)

		# Start timer
                timer = cv2.getTickCount()

                #cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                #cv2.waitKey(1)

		# Update tracker
                #ok, bbox = tracker.update(image)

		# Calculate Frames per second (FPS)
                #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);


		term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1) #cmsf
		

# Draw bounding box
                if ok == True:
                    #(x,y,w,h) = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))

		    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #cmsf
		    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) #cmsf

		    ret, track_window = cv2.CamShift(dst, track_window, term_crit) #cmsf
		    (x,y,w,h) = track_window #cmsf

                    #CX=int(bbox[0]+0.5*bbox[2]) #Center of X
                    #CY=int(bbox[1]+0.5*bbox[3])
                    S = w*h
                    
                    # Tracking success
                    p1 = (x, y)
                    p2 = (x + w, y + h)
                    cv2.rectangle(image, p1, p2, (255,0,0), 2, 1)
		    p10 = (x0, y0)
                    p20 = (x0 + w0, y0 + h0)
                    cv2.rectangle(image, p10, p20, (0,255,0), 2, 1)

                    d = L0 * m.sqrt(S / (w0*h0)) - L0
                    dx = x + w/2 - CX0
                    dy = y + h/2 - CY0
		    print("CX,CY,S,x,y,S0 =",int(x+0.5*w),int(y+0.5*h),S,x,y,w0*h0)
                    print(d,dx,dy)
		    
		    tracking(drone,d,dx,dy,L0)

		    
                else:
                # Tracking failure		    
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

		    #camshif--ref_https://qiita.com/MuAuan/items/a6e4aace2a6c0a7cb03d-----------------------------
		    #cap = cv2.VideoCapture(0)

		    track_window = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
    		    roi = image[y0:y0+h0, x0:x0+w0]

    		    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    		    img_mask = cv2.inRange(hsv_roi, numpy.array((0., 60.,32.)), numpy.array((180.,255.,255.)))

		    roi_hist = cv2.calcHist([hsv_roi], [0], img_mask, [180], [0,180])
    		    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        	    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
		    
		    ret,image = cap.read()
		    ok = True

		    #camshif_end--------------------------------------

		    #ok = tracker.init(image, bbox)
		    

		if frame.time_base < 1.0/60:
                    time_base = 1.0/60
                else:
                    time_base = frame.time_base
                frame_skip = int((time.time() - start_time)/time_base)

		#print(ok)

		    



#-------------------------------------------------for end
            break
	#print('stop fly')
                    

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
