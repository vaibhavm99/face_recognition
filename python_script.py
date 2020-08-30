import cv2
import face_recognition as fr
import sys

passport = sys.argv[1]
selfie = sys.argv[2]


try:
    img_passport = fr.load_image_file(passport)
    img_selfie = fr.load_image_file(selfie)
    img_passport = cv2.cvtColor(img_passport, cv2.COLOR_BGR2RGB)
    img_selfie = cv2.cvtColor(img_selfie, cv2.COLOR_BGR2RGB)
except:
    print("Wrong Path to the image")
    exit()


selfie_face_locs = fr.face_locations(img_selfie)
passport_face_loc = fr.face_locations(img_passport)[0]
encodes_selfie = fr.face_encodings(img_selfie)
encode_passport = fr.face_encodings(img_passport)[0]


cv2.rectangle(img_passport, (passport_face_loc[3],passport_face_loc[0]),(passport_face_loc[1],passport_face_loc[2]),(255,0,255),2)

for loc in selfie_face_locs:
    cv2.rectangle(img_selfie, (loc[3],loc[0]),(loc[1],loc[2]),(255,0,255),2)

lst = []

for encode in encodes_selfie:
    comparison = fr.compare_faces([encode_passport], encode)
    accuracy = fr.face_distance(encode_passport,encodes_selfie)
    accuracy = 1 - accuracy
    lst.append((comparison, accuracy))
flag = 0
for result, score in lst:
    if result[0] == True:
        print("IMAGE MATCHED")
        print(f"Confidence = {round(score[0], 4) * 100} %")
        flag = 1
if flag == 0:
    print("NO MATCH FOUND")
