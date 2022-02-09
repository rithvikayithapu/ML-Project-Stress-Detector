# ML-Project-Stress-Detector

<p>
    Stress and anxiety have a drastic impact on human wellbeing and health. With This pandemic people have developed the state of chronic stress, the human body and mind feels the anxiety of this by constantly developing the ways to recover from it and defend the mental and physical health from deteriorating Such stress responses can cause depression and affect the mental health. Moreover, excessive worrying and high anxiety can lead to even suicidal thoughts. The idea of the project is to analyse the working of a real-time non-intrusive monitoring system, which detects the emotional states of the person by considering facial expressions and emotional characteristics as parameters. This system classifies anger and disgust as stress related emotions. Stress and anxiety are psycho-somatic states being present as a side effect of modern, accelerated life rhythms. Stressors are perceived by the human body as threats, mobilizing all of its resources for defence. Frequent incidents of stress can even affect mental and physical health, due to increased heart rate. The detection of stress and anxiety in its early stages turns to be of great significance, especially if achieved without the excessive use of sensors and equipment or other monitoring equipment, which may cause extra stress and anxiety to the individual. The system has been built as a web app so that it can be accessed anywhere anytime. The dataset used for training has been obtained from Kaggle.
</p>

## How It Works

<ol>
    <li>OpenCV Video Capture has been used to grab the user's video.</li>
    <li>The frames are passed to a custom implementation of CNN. The CNN consists of 4 convolutional layers followed by a flatten layer and fully connected dense layer.</li>
    <li>The fully connected dense layer creates a one-dimensional vector which is passed to a Random Forest Classifier. The one dimensional vector contains scores for each of the seven emotions.(Angry, Disgust, Fear, Happy, Neutral, Sad and Surprise)</li>
    <li>The Random Forest Classifier is trained on a manually labelled dataset of images and classified the images into "Stressed", "Not Stressed" or "Don't Know".</li>
</ol>

## System Performance
![performance](https://github.com/rithvik2607/ML-Project-Stress-Detector/blob/master/images/performance.png)

## Website Images
![img1](https://github.com/rithvik2607/ML-Project-Stress-Detector/blob/master/images/web1.png)

![img2](https://github.com/rithvik2607/ML-Project-Stress-Detector/blob/master/images/web2.png)