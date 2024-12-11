# Tiêu đề ứng dụng
st.title("Breast Cancer Classification Demo")

# Giới thiệu
st.write("""
Ứng dụng này minh họa mô hình CNN phân loại ảnh chụp (có hoặc không có dấu hiệu ung thư).
""")

# Một số cài đặt
train_dir = "C:/data/train"
test_dir = "C:/data/test"
val_dir = "C:/data/valid"
classes = ["0", "1"]

# Kiểm tra xem thư mục có tồn tại không (trong môi trường Streamlit Cloud có thể không)
if not (os.path.exists(train_dir) and os.path.exists(test_dir) and os.path.exists(val_dir)):
    st.warning("Không tìm thấy dữ liệu trong các thư mục được chỉ định. Hãy kiểm tra đường dẫn hoặc upload dữ liệu.")
else:
    # Lựa chọn chạy tiền xử lý dữ liệu
    if st.button("Tiền xử lý dữ liệu"):
        # Train data
        st.write("Đang chuẩn bị dữ liệu train...")
        train_dataset = []
        for class_label in classes:
            class_path = os.path.join(train_dir, class_label)
            label_index = classes.index(class_label)
            for img_file in os.listdir(class_path):
                img_full_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_full_path)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    train_dataset.append([img, label_index])

        # Test data
        st.write("Đang chuẩn bị dữ liệu test...")
        test_dataset = []
        for class_label in classes:
            class_path = os.path.join(test_dir, class_label)
            label_index = classes.index(class_label)
            for img_file in os.listdir(class_path):
                img_full_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_full_path)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    test_dataset.append([img, label_index])

        # Validation data
        st.write("Đang chuẩn bị dữ liệu validation...")
        val_dataset = []
        for class_label in classes:
            class_path = os.path.join(val_dir, class_label)
            label_index = classes.index(class_label)
            for img_file in os.listdir(class_path):
                img_full_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_full_path)
                if img is not None:
                    img = cv2.resize(img, (224, 224))
                    val_dataset.append([img, label_index])

        # Shuffle
        random.shuffle(train_dataset)
        random.shuffle(test_dataset)
        random.shuffle(val_dataset)

        # Convert to numpy
        X_train_images = []
        y_train_labels = []
        for img, label in train_dataset:
            X_train_images.append(img)
            y_train_labels.append(label)
        X_train = np.array(X_train_images)/255.0
        y_train = np.array(y_train_labels)

        X_test_images = []
        y_test_labels = []
        for img, label in test_dataset:
            X_test_images.append(img)
            y_test_labels.append(label)
        X_test = np.array(X_test_images)/255.0
        y_test = np.array(y_test_labels)

        X_val_images = []
        y_val_labels = []
        for img, label in val_dataset:
            X_val_images.append(img)
            y_val_labels.append(label)
        X_val = np.array(X_val_images)/255.0
        y_val = np.array(y_val_labels)

        st.success("Dữ liệu đã sẵn sàng!")
        # Lưu vào session state để tránh load lại
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_val = X_val
        st.session_state.y_val = y_val

    # Hiển thị một số mẫu ảnh
    if 'X_train' in st.session_state:
        st.write("Hiển thị một số mẫu từ dữ liệu train:")
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        num_samples = min(6, len(X_train))
        random_samples = np.random.choice(len(X_train), num_samples, replace=False)
        fig, axes = plt.subplots(2, 3, figsize=(10, 7))
        for i, idx in enumerate(random_samples):
            ax = axes[i // 3, i % 3]
            ax.imshow(X_train[idx])
            ax.set_title(f"Label: {y_train[idx]}")
            ax.axis('off')
        st.pyplot(fig)

    # Train model
    if 'X_train' in st.session_state and st.button("Train Model"):
        # Lấy dữ liệu
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_val = st.session_state.X_val
        y_val = st.session_state.y_val

        # Định nghĩa model
        model = Sequential()
        model.add(Conv2D(100, (3, 3), activation="relu", input_shape=(224, 224, 3)))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(100, (3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(.3))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
        st.write(model.summary())

        # Train
        history = model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val), verbose=1)
        st.session_state.model = model
        st.session_state.history = history.history
        st.success("Đã train mô hình xong (với số epoch minh họa là 5)!")

    if 'history' in st.session_state:
        history = st.session_state.history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.plot(history['accuracy'], label='Train Accuracy')
        ax1.plot(history['val_accuracy'], label='Val Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(history['loss'], label='Train Loss')
        ax2.plot(history['val_loss'], label='Val Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        st.pyplot(fig)

    # Dự đoán trên test set
    if 'model' in st.session_state and 'X_test' in st.session_state and st.button("Evaluate on Test Set"):
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"Test Accuracy: {acc:.4f}")

        # Dự đoán và vẽ Confusion Matrix
        y_pred = (model.predict(X_test) >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-malignant', 'Malignant cancer'])
        disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # Classification Report
        report = classification_report(y_test, y_pred, target_names=['Non-malignant', 'Malignant cancer'])
        st.text("Classification Report:")
        st.text(report)

    # Cho phép người dùng upload ảnh để dự đoán
    if 'model' in st.session_state:
        st.write("Upload một ảnh để dự đoán:")
        uploaded_file = st.file_uploader("Chọn một file ảnh", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            img = Image.open(uploaded_file)
            st.image(img, caption='Ảnh upload', use_column_width=True)

            # Tiền xử lý ảnh
            img_resized = img.resize((224,224))
            img_array = image.img_to_array(img_resized)/255.0
            img_array = np.expand_dims(img_array, axis=0)

            model = st.session_state.model
            prediction = model.predict(img_array)
            label = "Cancer: Yes" if prediction[0]>=0.5 else "Cancer: No"
            st.write(f"Kết quả dự đoán: {label}")