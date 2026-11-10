import cv2
import winsound
import threading
from ultralytics import YOLO

stream_url = "http://10.207.97.209:81/stream"

person_model = YOLO("yolov8n.pt") 
weapon_model = YOLO(r"C:\Users\ravik\Downloads\Programs\project\best.pt") 

cap = cv2.VideoCapture(stream_url)
print("Intrusion & Weapon Detection System is starting... (Press 'q' to exit)")

is_alarming = False

def play_alarm():
    global is_alarming
    winsound.Beep(3000, 800) 
    is_alarming = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.resize(frame, (800, 600))
    
    person_results = person_model(frame, conf=0.5, verbose=False)
    weapon_results = weapon_model(frame, conf=0.5, verbose=False)
    
    threat_detected = False
    
    for r in person_results:
        for box in r.boxes:
            if person_model.names[int(box.cls[0])] == 'person':
                threat_detected = True
                
    for r in weapon_results:
        if len(r.boxes) > 0:
            threat_detected = True
            
    if threat_detected and not is_alarming:
        print("🚨 ALERT: INTRUSION OR WEAPON DETECTED!")
        is_alarming = True
        threading.Thread(target=play_alarm).start()
        
    if len(weapon_results[0].boxes) > 0:
        annotated_frame = weapon_results[0].plot()
    else:
        annotated_frame = person_results[0].plot()
        
    cv2.imshow("Ultimate Smart Security Camera", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()