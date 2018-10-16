package com.sashakhyzhun.opencvexample

import android.annotation.SuppressLint
import android.os.Bundle
import org.opencv.imgproc.Imgproc
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2
import android.view.View.OnTouchListener
import android.app.Activity
import android.util.Log
import android.view.*
import org.opencv.core.*


class MainActivity : Activity(), OnTouchListener, CvCameraViewListener2 {

    private var mIsColorSelected = false
    private var mRgba: Mat? = null
    private var mBlobColorRgba: Scalar? = null
    private var mBlobColorHsv: Scalar? = null
    private var mDetector: ColorBlobDetector? = null
    private var mSpectrum: Mat? = null
    private var SPECTRUM_SIZE: Size? = null
    private var CONTOUR_COLOR: Scalar? = null

    private lateinit var mat1: Mat
    private lateinit var mat2: Mat
    private lateinit var mat3: Mat

    private var mOpenCvCameraView: CameraBridgeViewBase? = null

    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    mOpenCvCameraView!!.enableView()
                    mOpenCvCameraView!!.setOnTouchListener(this@MainActivity)
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    init {
        Log.i(TAG, "Instantiated new " + this.javaClass)
    }

    /** Called when the activity is first created.  */
    public override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContentView(R.layout.activity_main)

        mOpenCvCameraView = findViewById(R.id.color_blob_detection_activity_surface_view)
        mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView!!.setCvCameraViewListener(this)

    }

    public override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    public override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    public override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        mRgba = Mat(height, width, CvType.CV_8UC4)
        mDetector = ColorBlobDetector()
        mSpectrum = Mat()
        mBlobColorRgba = Scalar(255.0)
        mBlobColorHsv = Scalar(255.0)
        SPECTRUM_SIZE = Size(200.0, 64.0)
        CONTOUR_COLOR = Scalar(255.0, 0.0, 0.0, 255.0)

        mat1 = Mat(width, height, CvType.CV_8UC4)
        mat2 = Mat(width, height, CvType.CV_8UC4)
        mat3 = Mat(width, height, CvType.CV_8UC4)
    }

    override fun onCameraViewStopped() {
        mRgba!!.release()
        mat1.release()
//        mat2.release()
//        mat3.release()
    }


    @SuppressLint("ClickableViewAccessibility")
    override fun onTouch(v: View, event: MotionEvent): Boolean {
        val cols = mRgba!!.cols()
        val rows = mRgba!!.rows()

        val xOffset = (mOpenCvCameraView!!.width - cols) / 2
        val yOffset = (mOpenCvCameraView!!.height - rows) / 2

        val x = event.x.toInt() - xOffset
        val y = event.y.toInt() - yOffset

        Log.i(TAG, "Touch image coordinates: ($x, $y)")

        if (x < 0 || y < 0 || x > cols || y > rows) return false

        val touchedRect = Rect()

        touchedRect.x = if (x > 4) x - 4 else 0
        touchedRect.y = if (y > 4) y - 4 else 0

        touchedRect.width = if (x + 4 < cols) x + 4 - touchedRect.x else cols - touchedRect.x
        touchedRect.height = if (y + 4 < rows) y + 4 - touchedRect.y else rows - touchedRect.y

        val touchedRegionRgba = mRgba!!.submat(touchedRect)

        val touchedRegionHsv = Mat()
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV_FULL)

        // Calculate average color of touched region
        mBlobColorHsv = Core.sumElems(touchedRegionHsv)
        val pointCount = touchedRect.width * touchedRect.height
        for (i in mBlobColorHsv!!.`val`.indices)
            mBlobColorHsv!!.`val`[i] /= pointCount.toDouble()

        mBlobColorRgba = convertScalarHsv2Rgba(mBlobColorHsv!!)

        Log.i(TAG, "Touched rgba color: (" + mBlobColorRgba!!.`val`[0] + ", " + mBlobColorRgba!!.`val`[1] +
                ", " + mBlobColorRgba!!.`val`[2] + ", " + mBlobColorRgba!!.`val`[3] + ")")

        mDetector!!.setHsvColor(mBlobColorHsv)

        Imgproc.resize(mDetector!!.getSpectrum(), mSpectrum!!, SPECTRUM_SIZE!!, 0.0, 0.0, Imgproc.INTER_LINEAR_EXACT)

        mIsColorSelected = true

        touchedRegionRgba.release()
        touchedRegionHsv.release()

        return false // don't need subsequent touch events
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat {
        mRgba = inputFrame.rgba()
        mat1 = inputFrame.rgba()

//        Core.transpose(mat1, mat2)
//        Imgproc.resize(mat2, mat3, mat3.size(), 0.0, 0.0, 0)
//        Core.flip(mat2, mat1, 1)

        Core.rotate(mat1, mat2, Core.ROTATE_90_COUNTERCLOCKWISE)

        if (mIsColorSelected) {
            mDetector!!.process(mRgba)
            val contours = mDetector!!.contours
            Log.e(TAG, "Contours count: " + contours.size)
            Imgproc.drawContours(mRgba!!, contours, -1, CONTOUR_COLOR!!)

            val colorLabel = mRgba!!.submat(4, 68, 4, 68)
            colorLabel.setTo(mBlobColorRgba)

            val spectrumLabel = mRgba!!.submat(4, 4 + mSpectrum!!.rows(), 70, 70 + mSpectrum!!.cols())
            mSpectrum!!.copyTo(spectrumLabel)
        }

        return mRgba!!
    }

    private fun convertScalarHsv2Rgba(hsvColor: Scalar): Scalar {
        val pointMatRgba = Mat()
        val pointMatHsv = Mat(1, 1, CvType.CV_8UC3, hsvColor)
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4)

        return Scalar(pointMatRgba.get(0, 0))
    }

    companion object {
        private val TAG = "MainActivity"
    }
}