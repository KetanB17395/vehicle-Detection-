import cv2
import numpy as np
from paddleocr import PaddleOCR,draw_ocr
import re
import pandas as pd

net = cv2.dnn.readNetFromONNX('vehdetyolov8n640.onnx')
ocr = PaddleOCR(use_angle_cls=True, lang='en')


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.5
Numberc = ['person','bicycle','car','motorcycle','bus','truck','traffic_light','stop_sign']


def append_to_excel(text,conf,  excel_file):

    # Load the existing Excel file (if it exists) or create a new one
    try:
        df = pd.read_excel(excel_file)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Vehicle Numbers','conf'])

    # Check if the cleaned text is not already in the DataFrame
    if text not in df['Vehicle Numbers'].values:
        # Append the cleaned text to the DataFrame
        new_data = pd.DataFrame({'Vehicle Numbers': [text],'conf':[conf]})
        df = pd.concat([df, new_data], ignore_index=True)

        # Save the DataFrame to the Excel file
        df.to_excel(excel_file, index=False)

def validate_ocr_word(ocr_word):
    army_vehicle_pattern = r'^[0-9A-Z]*?(\d{2}[A-Z][0-9]{3,6}[A-Z])[0-9A-Za-z]*?$'
    normal_vehicle_pattern = r'^[0-9A-Z]*?([A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4})[0-9A-Za-z]*?$'
    bharat_series_pattern = r'^[0-9A-Z]*?([2]\d{1}[BH]{2}\d{4}[A-Z]{1,2})[0-9A-Za-z]*?$'
    if re.match(army_vehicle_pattern, ocr_word):
        match = re.match(army_vehicle_pattern, ocr_word)
        word = match.group(1)[:]
        return "Army Vehicle: " + word
    elif re.match(normal_vehicle_pattern, ocr_word):
        match=re.match(normal_vehicle_pattern, ocr_word)
        word = match.group(1)[:]
        return "Normal Vehicle: " + word
    elif re.match(bharat_series_pattern, ocr_word):
        match=re.match(bharat_series_pattern, ocr_word)
        word=match.group(1)[:]
        return "Bharat Series Vehicle: " + word
    else:
        return None




cap = cv2.VideoCapture('/home/ketan/video_data_yuv_vehicle/crane.mp4') 
frame_index=0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_index%10==0:
        if frame is None:
            # Handle the case where the frame is None (e.g., end of the video)
            break


        if frame.shape[0] == 0:
            print(frame.shape)
            print(type(frame.shape))
            print(frame.shape[0])
            print(frame.shape[1])
            print("Skipping frame with zero height.")
            continue
        
        if frame.shape[1]==0:
            print("skipping the frame with 0 width")
            continue
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
        net.setInput(blob)
        preds = net.forward()
        
        preds = preds.transpose((0, 2, 1))
    
        class_ids, confs, boxes = list(), list(), list()
        
        image_height, image_width, _ = frame.shape
        x_factor = image_width / INPUT_WIDTH
        y_factor = image_height / INPUT_HEIGHT

        # Store cropped regions for OCR
        cropped_regions = []
        
        rows = preds[0].shape[0]
        
        for i in range(rows):
            row = preds[0][i]
            classes_score = row[4:]
            _, _, _, max_idx = cv2.minMaxLoc(classes_score)
            class_id = max_idx[1]
            classes=Numberc[class_id]
            conf=classes_score[class_id]
            if conf > CONFIDENCE_THRESHOLD and classes in ['car','motorcycle','bus','truck']:   
                
                confs.append(conf)
                
                class_ids.append(class_id)           
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)


                
        
            
        r_class_ids, r_confs, r_boxes = list(), list(), list()
        
        indexes = cv2.dnn.NMSBoxes(boxes, confs, SCORE_THRESHOLD, NMS_THRESHOLD)
        
            
        for i in indexes:
            #print(indexes)
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            #class_label=Numberc[class_ids[i]]

            #cropping the boxes
            cropped_region = frame[top:top+height, left:left+width]
            cropped_regions.append(cropped_region)

            label = "{}:{:.2f}".format(Numberc[class_ids[i]], confs[i])


            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 3)

            cv2.putText(frame, str(label), (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        frame = cv2.resize(frame, (800, 600))
        cv2.imshow('Video Detection', frame)

        for i, cropped_region in enumerate(cropped_regions):
            if cropped_region.shape[0] == 0:
                print("Skipping frame with zero height.")
                continue
            
            ocr_results = ocr.ocr(cropped_region)
           
            if ocr_results[0]:
             
                for line in ocr_results[0]:
                    for item in line:
                        if isinstance(item, tuple):
                            if len(item) == 2:
                                word, confidence = item
                                word = re.sub(r'[^A-Za-z0-9]', '', word)
                                result = validate_ocr_word(word)
                                if result:
                                    text_to_append=result
                                    excel_file_name='output.xlsx'
                                    append_to_excel(text_to_append,(confidence), excel_file_name)
                                
            else:
                print("No text found in the cropped region.")

       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_index= frame_index+1
cap.release()
cv2.destroyAllWindows()
