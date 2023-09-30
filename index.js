async function loadModel() {
    const model = await tf.loadLayersModel('model.json');

    // Get the input file element
    const inputElement = document.getElementById('image_input');

    // Listen for changes in the input
    inputElement.addEventListener('change', async (event) => {
        const file = event.target.files[0];

        if (file) {
            // Load and preprocess the selected image
            const imageElement = document.createElement('img');
            imageElement.src = URL.createObjectURL(file);
            await imageElement.decode();

            const canvas = document.createElement('canvas');
            canvas.width = 400;
            canvas.height = 400;

            const context = canvas.getContext('2d');
            context.drawImage(imageElement, 0, 0, 400, 400);

            const imageData = context.getImageData(0, 0, 400, 400);

            // Extract grayscale pixel values and normalize
            const data = new Float32Array(160000);
            for (let i = 0; i < imageData.data.length; i += 4) {
                const grayValue = (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3;
                data[i / 4] = grayValue / 255;
            }

            const tensor = tf.tensor4d(data, [1, 400, 400, 1]);

            // Make a prediction
            const prediction = model.predict(tensor);

            // Get the predicted class index
            const classIndex = prediction.argMax(1).dataSync()[0];

            // Define class names
            const classNames = ["Covid", "Normal", "Viral Pneumonia"];

            // Get the predicted class label
            const predictedClass = classNames[classIndex];
            prediction.print();
            // Print the predicted class to the console
            console.log("Predicted class:", predictedClass);

            // Update the HTML with the predicted class
            document.getElementById("prediction").innerHTML = predictedClass;
        }
    });
}

loadModel();
