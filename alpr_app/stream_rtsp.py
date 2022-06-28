import cv2
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
try:
# open any valid video stream(for e.g `foo1.mp4` file)
    cap= cv2.VideoCapture(0)
    stream = CamGear(source=0).start()
    output_params = {"-f": "rtsp", "-rtsp_transport":"tcp"}
    writer = WriteGear(output_filename='rtsp://localhost:8554/mystream', logging=True, **output_params)
    # loop over
    while True:
        # read frames from stream
        frame = stream.read()
        # check for frame if Nonetype
        if frame is None:
            break

        writer.write(frame)
        cv2.imshow("Output Frame", frame)

    cv2.destroyAllWindows()
    stream.stop()
    writer.close()

except Exception as error:
    print(error)