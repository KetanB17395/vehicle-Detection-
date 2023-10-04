
ocr_word='111111121BH1111DH123'

army_vehicle_pattern = r'^[0-9A-Z]*?(\d{2}[A-Z][0-9]{3,6}[A-Z])[0-9A-Za-z]*?$'
normal_vehicle_pattern = r'^[0-9A-Z]*?([A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4})[0-9A-Za-z]*?$'
bharat_series_pattern = r'^[0-9A-Z]*?([2]\d{1}[BH]{2}\d{4}[A-Z]{1,2})[0-9A-Za-z]*?$'
def validate_ocr_word(ocr_word):
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
    
import re


result = validate_ocr_word(ocr_word)
if result:
    print(result)
else:
    print('no_result')