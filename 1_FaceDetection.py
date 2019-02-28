"""
Face detect demo
"""
# import face_recognition
# import cv2
#
# img=face_recognition.load_image_file("./images/face.jpg")
# face_locations=face_recognition.face_locations(img)
#
# #显示图片
#
# img=cv2.imread('./images/face.jpg')
# cv2.namedWindow("OriginalPicture")
# cv2.imshow("OriginalPicture",img)
#
# #遍历每个人脸，并标注
# faceNum=len(face_locations)
# for i in range(0,faceNum):
#     top=face_locations[i][0]
#     right=face_locations[i][1]
#     bottom=face_locations[i][2]
#     left=face_locations[i][3]
#
#     start=(left,top)
#     end=(right,bottom)
#     color=(55,34,35)
#     thickness=3
#     cv2.rectangle(img,start,end,color,thickness)
#
#     cv2.namedWindow("FaceDetection")
#     cv2.imshow("FaceDetection",img)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

""""
案例二 识别图片中的人脸
"""

# import face_recognition
# Tom_Cruise_image=face_recognition.load_image_file("./images/Tom_Cruise.jpg")
# John_Salley_image=face_recognition.load_image_file("./images/John_Salley.jpg")
# test_image=face_recognition.load_image_file("./images/test.jpg")
#
# Tom_Cruise_encoding=face_recognition.face_encodings(Tom_Cruise_image)[0]
# John_Salley_encoding=face_recognition.face_encodings(John_Salley_image)[0]
# test_encoding=face_recognition.face_encodings(test_image)[0]
#
# results=face_recognition.compare_faces([Tom_Cruise_encoding,John_Salley_encoding],test_encoding)
# labels=['Tom_Cruise','John_Salley']
#
# print('results:'+str(results))
#
# for i in range(0,len(results)):
#     if results[i]==True:
#         print("The person is:"+labels[i])

"""
案例三  摄像头实时识别人脸
"""
import  face_recognition
import cv2

video_capture=cv2.VideoCapture(0)

John_Salley_img=face_recognition.load_image_file("./images/John_Salley.jpg")
John_Salley_face_encoding=face_recognition.face_encodings(John_Salley_img)[0]

face_locations=[]
face_encodings=[]
face_names=[]
process_this_frame=True

while True:
    ret,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)

    if process_this_frame:
        face_locations=face_recognition.face_locations(small_frame)
        face_encodings=face_recognition.face_encodings(small_frame,face_locations)

        face_names=[]
        for face_encoding in face_encodings:
            match=face_recognition.compare_faces([John_Salley_face_encoding],face_encoding)

            if match[0]:
                name="John_Salley"
            else:
                name="unknown"
            face_names.append(name)

    process_this_frame=not process_this_frame
    for (top,right,bottom,left),name in zip(face_locations,face_names):
        top*=4
        right*=4
        bottom*=4
        left*=4

        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)

        cv2.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,255),2)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(left+6,bottom-6),font,1.0,(255,255,255),1)

    cv2.imshow('Video',frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()



