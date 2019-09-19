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

    x = 200
    y = 200
    w = 224
    h = 224
    track_window=(x,y,w,h)
    # Reference Distance
    L0 = 100
    S0 = 50176 #224x224 #take#here.

    # Base Distance
    LB = 100
    # Define an initial bounding box
    bbox = (x, y, w, h)   #(287, 23, 86, 320)
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

        #drone.takeoff()
        #sleep(5)
        #drone.land()




        # skip first 300 frames
        frame_skip = 300
        while True:
#------------------------------------------for start
            for frame in container.decode(video=0):
		
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
		

		image = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)

		# Start timer
                timer = cv2.getTickCount()


                #start_time = time.time()

                #cv2.imshow('Canny', cv2.Canny(image, 100, 200))
                #cv2.waitKey(1)

                #if frame.time_base < 1.0/60:
                #    time_base = 1.0/60
                #else:
                #    time_base = frame.time_base
                #frame_skip = int((time.time() - start_time)/time_base)



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
                    bbox = cv2.selectROI(image, False)
                    print(bbox)
                    (x,y,w,h) = (int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
                    # Initialize tracker with first frame and bounding box
                    ok = tracker.init(image, bbox)



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
