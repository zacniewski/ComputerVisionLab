# import the necessary packages
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
import numpy as np
import urllib.request
import cv2
import os
from PIL import Image

# define the path to the face detector
#FACE_DETECTOR_PATH = "/home/artur/Downloads/opencv-3.0.0/data/haarcascades/haarcascade_frontalface_default.xml".\
#    format(base_path=os.path.abspath(os.path.dirname(__file__)))
FACE_DETECTOR_PATH = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml".\
    format(base_path=os.path.abspath(os.path.dirname(__file__)))


@csrf_exempt
def detect_faces(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("image_url", None)
            #url = "http://whychess.com/sites/default/files/imagecache/image_full_node/botvinnik.JPEG"
            #url = "http://chesshive.com/wp-content/uploads/2016/05/bobby-fischer-anatoly-karpov.jpg"

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)

        # convert the image to grayscale, load the face cascade detector,
        # and detect faces in the image
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
        rects = detector.detectMultiScale(image,
                                          scaleFactor=1.1, minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
        # flags = cv2.CASCADE_SCALE_IMAGE for OpenCV 3.x
        # flags=cv2.cv.CV_HAAR_SCALE_IMAGE for OpenCV 2.4.x

        # construct a list of bounding boxes from the detection
        rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

        # update the data dictionary with the faces detected
        data.update({"number_of_faces": len(rects), "regions_of_faces": rects, "success": True})

    # loop over the faces and draw them on the image
    for (startX, startY, endX, endY) in data["regions_of_faces"]:
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show image
    #cv2.imshow("Your image", image)
    #cv2.waitKey(0)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/img/cv/df1.jpg', image_rgb)
    return render(request, 'vision/detect_faces.html')


def _grab_image(path=None, stream=None, url=None):
    # if the path is not None, then load the image from disk
    if path is not None:
        image = cv2.imread(path)

    # otherwise, the image does not reside on disk
    else:
        # if the URL is not None, then download the image
        if url is not None:
            resp = urllib.request.urlopen(url)
            data = resp.read()

        # if the stream is not None, then the image has been uploaded
        elif stream is not None:
            data = stream.read()

        # convert the image to a NumPy array and then read it into
        # OpenCV format
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image


def submit_image_with_faces(request):
    #return render(request, 'vision/submit_image_with_faces.html')
    return render(request, 'vision/submit_image_with_faces.html')


def vision_lab(request):
    return render(request, 'vision/VisionLabHomePage.html')


def computer_vision_algorithms(request):
    return render(request, 'vision/computer_vision_algorithms.html')


def detect_color_with_camera(request):
    return render(request, 'vision/detect_color_with_camera.html')


def detect_face_with_camera(request):
    return render(request, 'vision/detect_face_with_camera.html')


def detect_face_with_camera_clown(request):
    return render(request, 'vision/detect_face_with_camera_clown.html')


def detect_face_with_camera_anonymous(request):
    return render(request, 'vision/detect_face_with_camera_anonymous.html')


def detect_face_with_camera_bear(request):
    return render(request, 'vision/detect_face_with_camera_bear.html')


def detect_face_with_camera_politician(request):
    return render(request, 'vision/detect_face_with_camera_politician.html')


#######################################################################
#######################################################################


@csrf_exempt
def thresholding(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("image_url", None)
            thr = request.POST.get("threshold")

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(image,5)
        # here we can specify our 'thr'
        ret,th1 = cv2.threshold(img, float(thr), 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(img, float(thr), cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,11,2)
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                  'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]
        #cv2.imshow('Thresholded Image', images[3])
        #cv2.waitKey(0)

    #image_rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/img/cv/th_original.jpg', image)
    cv2.imwrite('static/img/cv/th_blurred.jpg', img)
    #image_rgb_th1 = cv2.cvtColor(th1, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/img/cv/th1.jpg', th1)
    #image_rgb_th2 = cv2.cvtColor(th2, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/img/cv/th2.jpg', th2)
    #image_rgb_th3 = cv2.cvtColor(th3, cv2.COLOR_BGR2RGB)
    cv2.imwrite('static/img/cv/th3.jpg', th3)
    return render(request, 'vision/thresholding.html', {'thr': thr})


def submit_image_with_threshold(request):
    return render(request, 'vision/submit_image_with_threshold.html')

#######################################################################
#######################################################################


@csrf_exempt
def canny(request, thr1, thr2):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("image_url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)
            cv2.imwrite('static/img/cv/canny_original.jpg', image)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(image, float(thr1), float(thr2))

        #cv2.imshow('Canny Image', edges)
        #cv2.waitKey(0)
    else:
        # when GET request  -changing thresholds manually
        image = cv2.imread('static/img/cv/canny_original.jpg')
        edges = cv2.Canny(image, float(thr1), float(thr2))
        cv2.imwrite('static/img/cv/canny_edges.jpg', edges)
        return render(request, 'vision/canny_edges.html', {'thr1': thr1, 'thr2': thr2})

    cv2.imwrite('static/img/cv/canny_edges.jpg', edges)
    return render(request, 'vision/canny_edges.html', {'thr1': thr1, 'thr2': thr2})


def submit_canny(request):
    return render(request, 'vision/submit_image_canny.html')


#######################################################################
#######################################################################


@csrf_exempt
def keypoints(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("image_url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Initiate ORB object
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    keypoints = orb.detect(gray_image, None)
    # compute the descriptors with ORB
    keypoints, descriptors = orb.compute(gray_image, keypoints)
    # draw only the location of the keypoints without size or     orientation
    final_keypoints = cv2.drawKeypoints(image, keypoints, None,
                                        color=(0, 255, 0), flags=0)
    cv2.imwrite('static/img/cv/keypoints.jpg', final_keypoints)

    return render(request, 'vision/keypoints.html')


def submit_image_to_keypoints(request):
    return render(request, 'vision/submit_image_keypoints.html')

#######################################################################
#######################################################################


@csrf_exempt
def coins(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("image_url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (11, 11), 0)

    # The first thing we are going to do is apply edge detection to
    # the image to reveal the outlines of the coins
    edged = cv2.Canny(blurred, 30, 150)

    # Find contours in the edged image.
    # NOTE: The cv2.findContours method is DESTRUCTIVE to the image
    # you pass in. If you intend on reusing your edged image, be
    # sure to copy it before calling cv2.findContours
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # How many contours did we find?
    number_of_coins = len(cnts)

    # Let's highlight the coins in the original image by drawing a
    # green circle around them
    coins = image.copy()
    cv2.drawContours(coins, cnts, -1, (0, 255, 0), 2)
    cv2.imwrite('static/img/cv/coins.jpg', coins)

    return render(request, 'vision/coins.html', {'number_of_coins': number_of_coins})


def submit_image_to_coins(request):
    return render(request, 'vision/submit_image_coins.html')

#######################################################################
#######################################################################


@csrf_exempt
def segmentation(request):
    # initialize the data dictionary to be returned by the request
    data = {"success": False}

    # check to see if this is a post request
    if request.method == "POST":
        # check to see if an image was uploaded
        if request.FILES.get("image", None) is not None:
            # grab the uploaded image
            image = _grab_image(stream=request.FILES["image"])

        # otherwise, assume that a URL was passed in
        else:
            # grab the URL from the request
            url = request.POST.get("image_url", None)

            # if the URL is None, then return an error
            if url is None:
                data["error"] = "No URL provided."
                return JsonResponse(data)

            # load the image and convert
            image = _grab_image(url=url)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/img/cv/before_segmentation.jpg", image)
    ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imwrite('static/img/cv/sure_bg.jpg', sure_bg)


    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    cv2.imwrite('static/img/cv/sure_fg.jpg', sure_fg)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)
    cv2.imwrite('static/img/cv/markers.jpg', markers)

    image[markers == -1] = [255, 0, 0]

    cv2.imwrite('static/img/cv/segmentation.jpg', image)

    return render(request, 'vision/segmentation.html')


def submit_image_to_segmentation(request):
    return render(request, 'vision/submit_image_segmentation.html')

#######################################################################
#######################################################################


def about(request):
    return render(request, 'vision/about.html')


def authors(request):
    return render(request, 'vision/authors.html')
