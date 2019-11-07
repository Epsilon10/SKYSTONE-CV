package org.firstinspires.ftc.teamcode;

import android.graphics.Bitmap;

import com.acmerobotics.dashboard.FtcDashboard;
import com.acmerobotics.dashboard.config.Config;
import com.acmerobotics.dashboard.telemetry.TelemetryPacket;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.eventloop.opmode.TeleOp;
import com.vuforia.Image;
import com.vuforia.PIXEL_FORMAT;
import com.vuforia.Vuforia;

import org.firstinspires.ftc.robotcore.external.ClassFactory;
import org.firstinspires.ftc.robotcore.external.hardware.camera.WebcamName;
import org.firstinspires.ftc.robotcore.external.matrices.OpenGLMatrix;
import org.firstinspires.ftc.robotcore.external.navigation.VuforiaLocalizer;
import org.firstinspires.ftc.teamcode.vision.VisionPipeline;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.openftc.easyopencv.OpenCvCamera;
import org.openftc.easyopencv.OpenCvCameraRotation;
import org.openftc.easyopencv.OpenCvWebcam;

import java.util.concurrent.BlockingQueue;

import static org.firstinspires.ftc.robotcore.external.navigation.VuforiaLocalizer.CameraDirection.BACK;

@Config
@TeleOp(name="VisionTest")
public class VisionTest extends LinearOpMode {
    private static final VuforiaLocalizer.CameraDirection CAMERA_CHOICE = BACK;
    private static final boolean PHONE_IS_PORTRAIT = false  ;

    /*
     * IMPORTANT: You need to obtain your own license key to use Vuforia. The string below with which
     * 'parameters.vuforiaLicenseKey' is initialized is for illustration only, and will not function.
     * A Vuforia 'Development' license key, can be obtained free of charge from the Vuforia developer
     * web site at https://developer.vuforia.com/license-manager.
     *
     * Vuforia license keys are always 380 characters long, and look as if they contain mostly
     * random data. As an example, here is a example of a fragment of a valid key:
     *      ... yIgIzTqZ4mWjk9wd3cZO9T1axEqzuhxoGlfOOI2dRzKS4T0hQ8kT ...
     * Once you've obtained a license key, copy the string from the Vuforia web site
     * and paste it in to your code on the next line, between the double quotes.
     */
    private static final String VUFORIA_KEY =
            "AT8pGt3/////AAABmQ9LKBWthkikgQSErtn4C1GN+U/k35mErGuydnhrXtBLs2+wEnRYzMx2qJC0Q+4bHLUaWRZ18gRQcTZOoaKDYfG7yIcNfsexI4G5IdAgwfAZnSbrWco7IW2mdaHZrQ5mw/u0mh1RHbcPdK3JAheEknMP53n73JNNBFbEcB+IN2qPSI4AUrWqK3TuAl7XCnEBQrHKB7kU62rXWs+4r4/RcNB0g/yMZ3S5Yv7vfHYGMEA3/Wj+4PC/6v/pO9StgMjxKaVZMjTYiHvUN6yi6CgVfQlKlmkEMU0IR60PcUgA9hKq9CPXVNPN1tXCTGFGdd+WbhFEGdkbZxY3scU85G4kDQy2oNbFRaClpdHYINBOV1U1";

    // Since ImageTarget trackables use mm to specifiy their dimensions, we must use mm for all the physical dimension.
    // We will define some constants and conversions here
    private static final float mmPerInch        = 25.4f;
    private static final float mmTargetHeight   = (6) * mmPerInch;          // the height of the center of the target image above the floor

    // Constant for Stone Target
    private static final float stoneZ = 2.00f * mmPerInch;

    // Constants for the center support targets
    private static final float bridgeZ = 6.42f * mmPerInch;
    private static final float bridgeY = 23 * mmPerInch;
    private static final float bridgeX = 5.18f * mmPerInch;
    private static final float bridgeRotY = 59;                                 // Units are degrees
    private static final float bridgeRotZ = 180;

    // Constants for perimeter targets
    private static final float halfField = 72 * mmPerInch;
    private static final float quadField  = 36 * mmPerInch;

    // Class Members
    private OpenGLMatrix lastLocation = null;
    private VuforiaLocalizer vuforia = null;

    /**
     * This is the webcam we are to use. As with other hardware devices such as motors and
     * servos, this device is identified using the robot configuration tool in the FTC application.
     */
    WebcamName webcamName = null;

    private boolean targetVisible = false;
    private float phoneXRotate    = 0;
    private float phoneYRotate    = 0;
    private float phoneZRotate    = 0;

    VisionPipeline p;

    private boolean shouldWrite = false;
    @Override
    public void runOpMode() {
        p = new VisionPipeline();
        // webcamName = hardwareMap.get(WebcamName.class, "Webcam 1");

        /*
         * Configure Vuforia by creating a Parameter object, and passing it to the Vuforia engine.
         * We can pass Vuforia the handle to a camera preview resource (on the RC phone);
         * If no camera monitor is desired, use the parameter-less constructor instead (commented out below).
         */
        int cameraMonitorViewId = hardwareMap.appContext.getResources().getIdentifier("cameraMonitorViewId", "id", hardwareMap.appContext.getPackageName());
        VuforiaLocalizer.Parameters parameters = new VuforiaLocalizer.Parameters(cameraMonitorViewId);

        // VuforiaLocalizer.Parameters parameters = new VuforiaLocalizer.Parameters();

        parameters.vuforiaLicenseKey = VUFORIA_KEY;

        /**
         * We also indicate which camera on the RC we wish to use.
         */
        // parameters.cameraName = webcamName;

        //  Instantiate the Vuforia engine
        vuforia = ClassFactory.getInstance().createVuforia(parameters);

        Vuforia.setFrameFormat(PIXEL_FORMAT.RGB565, true);
        vuforia.setFrameQueueCapacity(1);

        FtcDashboard dashboard = FtcDashboard.getInstance();

        Mat frame;
        waitForStart();

        while (opModeIsActive()) {
            if (gamepad1.a)
                shouldWrite = true;
            else
                shouldWrite = false;
            VuforiaLocalizer.CloseableFrame vuFrame = null;
            if (!vuforia.getFrameQueue().isEmpty()) {
                try {
                    vuFrame = vuforia.getFrameQueue().take();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }

                if (vuFrame == null) continue;

                for (int i = 0; i < vuFrame.getNumImages(); i++) {
                    Image img = vuFrame.getImage(i);
                    if (img.getFormat() == PIXEL_FORMAT.RGB565) {
                        Bitmap bm = Bitmap.createBitmap(img.getWidth(), img.getHeight(), Bitmap.Config.RGB_565);
                        bm.copyPixelsFromBuffer(img.getPixels());
                        Mat mat = bitmapToMat(bm, CvType.CV_8UC3);
                        Mat ret = p.processFrame(mat);
                        Bitmap displayBitmap = Bitmap.createBitmap(ret.width(), ret.height(), Bitmap.Config.RGB_565);
                        Utils.matToBitmap(ret, displayBitmap);
                        dashboard.sendImage(displayBitmap);
                    }
                }
            }
            dashboard.sendTelemetryPacket(new TelemetryPacket());
            telemetry.addData("gfsdafdsy: ",p.getVumarkLeftBoundary());
            telemetry.update();
        }

    }

    private Mat bitmapToMat (Bitmap bit, int cvType) {
        Mat newMat = new Mat(bit.getHeight(), bit.getWidth(), cvType);

        Utils.bitmapToMat(bit, newMat);

        return newMat;
    }

}
