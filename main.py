import tkinter as tk  # For GUI
from tkinter import filedialog
import pickle  # For binary file persistance storage
import zipfile as zf  # For combining the binary files into one file
import os
from classifier import DrawingClassifier as dClf
import plots


class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master, relief="raised", bd=2)
        self.master = master

        # Initalize variables
        self.current_num_of_features = 0
        self.each_num_of_features = []
        self.num_of_drawings = 0
        self.labels = list()
        self.magnitudes = list()
        self.cartesian_cords = list()
        self.drawing_ovals = [[]]
        self.drawing_labels = list()
        self.drawing_magnitudes = list()
        self.drawing_cartesian_cords = [[]]

        # Create and place widgets
        self.pack()
        self.create_widgets()

        # Bind mouse button to draw on click
        self.canvas.bind("<B1-Motion>", self.paint)

        # Create toplevel menu
        menubar = tk.Menu(master)
        menubar.add_command(label="Clear Canvas", command=self.clearCanvas)
        menubar.add_separator()
        menubar.add_command(label="View Drawings", command=self.viewDrawings)
        menubar.add_separator()

        # Save, load, and reset dropdown options
        session_menu = tk.Menu(master)
        session_menu.add_command(label="Save Session", command=self.saveSession)
        session_menu.add_command(label="Load Session", command=self.loadSession)
        session_menu.add_separator()
        session_menu.add_command(label="Reset Session", command=self.resetSession)

        # Save, load, and reset dropdown options
        performance_menu = tk.Menu(master)
        performance_menu.add_command(label="View Performance Window", command=self.viewPerformance)
        performance_menu.add_command(label="Plot SVM kernels for Caresian method", command=self.SVMPLots)

        menubar.add_cascade(label="Performance", menu=performance_menu)
        menubar.add_cascade(label="Session", menu=session_menu)
        master.config(menu=menubar)

        seperator = tk.Frame(master, height=2, bg="grey")
        seperator.pack(fill="both", expand=True)

    def create_widgets(self):
        # Create canvas for drawing
        self.canvas_frame = tk.Frame()
        self.canvas_frame.pack(fill="x", expand="true")
        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(expand="true")

        # Create frame to add a new label through entry and button
        self.new_label_frame = tk.Frame()
        self.new_label_frame.pack(fill="x", expand="true", pady=2)
        self.new_label_text = tk.StringVar()
        self.new_label = tk.Entry(self.new_label_frame, textvariable=self.new_label_text)
        self.new_label.pack(side="left", expand="true")
        self.new_label_text.set("Type of drawing?")
        self.add_new_label = tk.Button(self.new_label_frame, text="Add Drawing As New Label", command=self.addNewLabel)
        self.add_new_label.pack(side="left", expand="true")

        # Create frame to add drawing to existijng label through optionmenu and button
        self.existing_label_frame = tk.Frame()
        self.existing_label_frame.pack(fill="x", expand="true", pady=2)
        self.existing_label_text = tk.StringVar()
        self.existing_label_text.set("Currently No Labels")
        self.existing_labels = tk.OptionMenu(self.existing_label_frame, self.existing_label_text, "")
        self.existing_labels.pack(side="left", expand="true")
        self.add_existing_label = tk.Button(self.existing_label_frame, text="Add Drawing As Existing Label", command=self.addExistingLabel)
        self.add_existing_label.pack(side="left", expand="true")

        # Create frame to classify through button and to display results through label
        self.classification_frame = tk.Frame()
        self.classification_frame.pack(fill="x", expand="true", pady=10)
        self.classify_button = tk.Button(self.classification_frame, text="Classify Drawing Using Magnitudes Method", command=self.classifyDrawing)
        self.classify_button.pack(expand="true")
        self.classify_cords_button = tk.Button(self.classification_frame, text="Classify Drawing Using Cartesian Method", command=self.classifyDrawingUsingCartesianMethod)
        self.classify_cords_button.pack(expand="true")
        self.classify_results_text = tk.StringVar()
        self.classify_results = tk.Label(self.classification_frame, textvariable=self.classify_results_text)
        self.classify_results_text.set("Press and drag mouse on canvas to draw")
        self.classify_results.pack(side="bottom")

    def paint(self, event):
        ''' Draws tiny ovals representing pixels on the canvas'''

        # Get posion of mouse clicks
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        mag = (x1 ** 2 + y1 ** 2) ** .5

        # Save point data for classification and display purposes
        self.drawing_ovals[self.num_of_drawings].append([x1, y1, x2, y2])
        self.drawing_cartesian_cords[self.num_of_drawings].append([x1, y1])
        self.current_num_of_features += 1
        self.magnitudes.append(mag)

        # Display oval on canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill="black")

    def clearCanvas(self):
        ''' Clears all points on the canvas '''
        self.canvas.delete("all")  # Clear Drawing
        self.drawing_ovals[self.num_of_drawings] = []
        self.magnitudes = []
        del self.drawing_cartesian_cords[-1]
        self.drawing_cartesian_cords.append([])

    def addNewLabel(self):
        ''' Adds new label for classifier '''
        # Get user label entry
        new_label = self.new_label_text.get()

        # If label already exists in existing labels, don't add it as a new label
        if new_label in self.labels:
            # Add drawing as a training example using existing label
            self.drawing_magnitudes.append(self.magnitudes)
            self.drawing_labels.append(new_label)

        # Add as existing label
        else:
            # Add drawing as a training example using new label
            self.labels.append(new_label)

            # Clear label text field
            self.new_label.delete(0, 'end')

            # Update exising labels dropdown menu
            self.updateOptionsMenu(self.existing_labels, self.existing_label_text, self.labels, "Choose an existing label")

            # Add drawing as a training example
            self.drawing_magnitudes.append(self.magnitudes)
            self.drawing_labels.append(new_label)

        # Empty points list to ready for next drawing
        self.magnitudes = list()

        # Clear drawing
        self.canvas.delete("all")

        self.num_of_drawings += 1
        self.drawing_ovals.append([])
        self.drawing_cartesian_cords.append([])
        self.each_num_of_features.append(self.current_num_of_features)
        self.current_num_of_features = 0

    def addExistingLabel(self):
        ''' Adds label for classification that is already in labels list  '''
        # Get label
        label = self.existing_label_text.get()

        # Add drawing as a training example
        self.drawing_magnitudes.append(self.magnitudes)
        self.drawing_labels.append(label)

        # Empty points list to ready for next drawing
        self.magnitudes = list()

        # Clear drawing
        self.canvas.delete("all")

        self.num_of_drawings += 1
        self.drawing_ovals.append([])
        self.drawing_cartesian_cords.append([])
        self.each_num_of_features.append(self.current_num_of_features)
        self.current_num_of_features = 0

    def classifyDrawing(self):
        ''' Clasifies drawing on canvas using magnitudes and poly kernel '''
        # Declare classifier
        if (self.drawing_cartesian_cords[-1] == []):
            self.classify_results_text.set("There is nothing to classify. Please draw something.")
            return

        clf = dClf()

        # Training requires the labelled exampels and the unlabelled drawing
        clf.train(self.drawing_magnitudes, self.drawing_labels, self.magnitudes)

        # Predict new drawing using the trained classifier
        classified_label = clf.predict(self.magnitudes)

        # Output label of drawing
        self.classify_results_text.set(classified_label)

    def classifyDrawingUsingCartesianMethod(self):
        ''' Clasifies drawing on canvas using cartesian cordinates and linear kernel '''
        if (self.drawing_cartesian_cords[-1] == []):
            self.classify_results_text.set("There is nothing to classify. Please draw something.")
            return

        # Declare classifier
        clf = dClf()

        cartesian_cords = self.drawing_cartesian_cords

        # Training requires the labelled exampels and the unlabelled drawing
        classified_label = clf.ClassifyCaresianMethod(cartesian_cords, self.drawing_labels)
        # Output label of drawing
        self.classify_results_text.set("Classifier claims drawing is: " + str(classified_label))

    def viewPerformance(self):
        self.performance_window = tk.Toplevel()
        feature_type_select = tk.Label(self.performance_window, text="Feature Representations").grid(row=0, column=0, sticky="W")
        self.feature_type_var = tk.StringVar()
        mag_feature_type_radio = tk.Radiobutton(self.performance_window, var=self.feature_type_var, text="As Magnitudes", value="mags").grid(row=1, column=0, sticky="W")
        cord_feature_type_radio = tk.Radiobutton(self.performance_window, var=self.feature_type_var, text="As Cartesian Cords", value="cords").grid(row=2, column=0, sticky="W")
        self.feature_type_var.set("mags")

        self.sorted_checkbtn_var = tk.IntVar()
        sorted_checkbtn = tk.Checkbutton(self.performance_window, text="Sort Magnitudes", variable=self.sorted_checkbtn_var).grid(row=1, column=0)

        seperator2 = tk.Label(self.performance_window, text="").grid(row=5, column=0)

        feature_type_select = tk.Label(self.performance_window, text="Percent of data to be split for training?").grid(row=6, column=0, sticky="W")
        self.test_train_split_var = tk.StringVar()
        test_train_split_entry = tk.Entry(self.performance_window, textvariable=self.test_train_split_var, width="5").grid(row=7, column=0)
        self.test_train_split_var.set("20")

        seperator3 = tk.Label(self.performance_window, text="").grid(row=8, column=0)

        self.thorough_checkbtn_var = tk.IntVar()
        thorough_checkbtn = tk.Checkbutton(self.performance_window, text="Average accuracy of each test-train split?", variable=self.thorough_checkbtn_var).grid(row=9, column=0, sticky="W")

        seperator4 = tk.Label(self.performance_window, text="").grid(row=10, column=0)

        self.kernel_text = tk.StringVar()
        self.kernel_text.set("Select SVM kernel")
        kernel_menu = tk.OptionMenu(self.performance_window, self.kernel_text, "poly", "rbf", "linear").grid(row=11, column=0)

        seperator5 = tk.Label(self.performance_window, text="").grid(row=12, column=0)

        begin_text_btn = tk.Button(self.performance_window, text="Begin Performance Test", command=self.performanceTest).grid(row=13, column=0)

    def performanceTest(self):
        ''' Tests accuracy using passed parameters and displays results in text box '''

        if (self.num_of_drawings <= 3):
            self.classify_results_text.set("Need at least 4 examples to text accuracy")
            return

        feature_type = self.feature_type_var.get()  # magnitudes or cartesian cordinates
        areSorted = (self.sorted_checkbtn_var.get() == 1)  # true or false
        percent_split = int(self.test_train_split_var.get())  # 1-99 percent
        isThorough = (self.thorough_checkbtn_var.get() == 1)  # true or false
        kernel_type = self.kernel_text.get()  # poly, rbf, or linear

        self.clf = dClf()
        accuracy = "Error"

        if (sorted(list(feature_type)) == sorted(list("mags"))):
            if (areSorted):
                sorted_magnitudes = []
                for mags in self.drawing_magnitudes:
                    sorted_magnitudes.append(sorted(mags))
                accuracy = self.clf.getAccuracy(sorted_magnitudes, self.drawing_labels, kernel_type, percent_split, isThorough)
            else:
                accuracy = self.clf.getAccuracy(self.drawing_magnitudes, self.drawing_labels, kernel_type, percent_split, isThorough)

        else:
            if (sorted(list(kernel_type)) == sorted(list("poly"))):
                self.classify_results_text.set("Polynomial kernel not supported for cordinate method")
                return
            if (self.drawing_cartesian_cords[-1] == []):
                    del self.drawing_cartesian_cords[-1]
            accuracy = self.clf.getAccuracyCartesianMethod(self.drawing_cartesian_cords, self.drawing_labels, kernel_type, percent_split, isThorough)
            self.drawing_cartesian_cords.append([])

        self.classify_results_text.set("Classifier accuracy on your drawings is " + accuracy + "%")

    def SVMPLots(self):
        ''' Plots one example from each label, comparing poly and rbf kernels'''
        plots.plot(self.drawing_cartesian_cords, self.drawing_cartesian_labels, self.labels)

    def viewDrawings(self):
        self.view_window = tk.Toplevel()
        prompt_label = tk.Label(self.view_window, text="View Drawing")
        prompt_label.pack()
        self.drawings_text = tk.StringVar()
        self.drawings_text.set("No Drawings Available")
        self.drawings_text.trace("w", self.viewDrawingsOnSelected)
        self.drawings_select = tk.OptionMenu(self.view_window, self.drawings_text, "")
        self.drawings_select.pack()
        self.delete_drawing_btn = tk.Button(self.view_window, text="Delete Drawing", command=self.deleteDrawing)
        self.delete_drawing_btn.pack(side="bottom")
        self.delete_drawing_btn['state'] = 'disabled'
        if (len(self.drawing_labels) > 0):
            self.drawing_names = list()
            # Populate dropdown menu names of each drawing
            for i in range(len(self.drawing_labels)):
                self.drawing_names.append(str(i + 1) + ": feats(" + str(len(self.drawing_ovals[i])) + ")_" + self.drawing_labels[i])

            self.updateOptionsMenu(self.drawings_select, self.drawings_text, self.drawing_names, "Choose Drawing")
        self.view_canvas = tk.Canvas(self.view_window)
        self.view_canvas.pack()

    def viewDrawingsOnSelected(self, *args):
        if (sorted(list(self.drawings_text.get())) != sorted(list("No Drawings Available")) and
           sorted(list(self.drawings_text.get())) != sorted(list("Choose Drawing")) and
           sorted(list(self.drawings_text.get())) != sorted(list(""))):
            # Clear drawing
            self.view_canvas.delete("all")
            # Create new drawing on canvas
            index = self.drawing_names.index(self.drawings_text.get())
            for oval in self.drawing_ovals[index]:
                self.view_canvas.create_oval(oval[0], oval[1], oval[2], oval[3], fill="black")
            # Get current drawing and enable button for delete drawing option
            self.current_drawing_index = index
            self.delete_drawing_btn['state'] = 'normal'

    def deleteDrawing(self):
        # Clear drawing
        self.view_canvas.delete("all")
        # Remove various representations of drawing
        del self.drawing_ovals[self.current_drawing_index]
        del self.drawing_magnitudes[self.current_drawing_index]
        del self.drawing_cartesian_cords[self.current_drawing_index]
        del self.drawing_labels[self.current_drawing_index]
        self.num_of_drawings -= 1
        # Get removed name and convert it to its label text
        removed_name = self.drawing_names.pop(self.current_drawing_index)
        removed_name = removed_name[(removed_name.index(")_") + 2):]
        print(removed_name)
        # Update labels if no more drawings in the label exists after deletion
        still_exists = False
        for name in self.drawing_names:
            if (name is removed_name):
                still_exists = True
        if (not still_exists):
            del self.labels[self.labels.index(removed_name)]

        # Update dropdown menu names
        if (len(self.drawing_labels) > 0):
            self.drawing_names = list()
            for i in range(len(self.drawing_labels)):
                self.drawing_names.append(str(i + 1) + ": feats(" + str(len(self.drawing_ovals[i])) + ")_" + self.drawing_labels[i])
            self.updateOptionsMenu(self.drawings_select, self.drawings_text, self.drawing_names, "Choose Drawing")
        else:
            self.updateOptionsMenu(self.drawings_select, self.drawings_text, [], "No Drawings Available")

    def saveSession(self):
        # Update status
        self.classify_results_text.set("Saving Session")

        session_name = filedialog.asksaveasfilename(initialdir="./saved/", title="Select where to save session", filetypes=(("session", "*.session"), ("all files", "*.*")))

        # Create zip file to contain the saved data
        self.zip = zf.ZipFile(session_name + ".session", "w")

        # Save every variable
        self.save(self.labels, "labels")
        self.save(self.magnitudes, "magnitudes")
        self.save(self.drawing_ovals, "drawing_ovals")
        self.save(self.drawing_labels, "drawing_labels")
        self.save(self.drawing_magnitudes, "drawing_magnitudes")
        self.save(self.num_of_drawings, "num_of_drawings")
        self.save(self.drawing_cartesian_cords, "drawing_cartesian_cords")

        self.zip.close()

        # Update success status
        self.classify_results_text.set("Session saved successfully.")

    def loadSession(self):
        # Update status
        self.classify_results_text.set("Loading Session")

        session_name = filedialog.askopenfilename(initialdir="./saved/", title="Select session file to load from", filetypes=(("session", "*.session"), ("all files", "*.*")))

        # Open zip file to containing the saved data
        self.zip = zf.ZipFile(session_name, "r")

        # Load every variable
        self.labels = self.load("labels")
        self.magnitudes = self.load("magnitudes")
        self.drawing_ovals = self.load("drawing_ovals")
        self.drawing_labels = self.load("drawing_labels")
        self.drawing_magnitudes = self.load("drawing_magnitudes")
        self.num_of_drawings = self.load("num_of_drawings")
        self.drawing_cartesian_cords = self.load("drawing_cartesian_cords")

        self.zip.close()

        # Update success status and options menu
        self.updateOptionsMenu(self.existing_labels, self.existing_label_text, self.labels, "Choose an existing label")
        self.classify_results_text.set("Session loaded successfully.")

        # Clear drawing
        self.canvas.delete("all")
        self.current_num_of_features = 0

    def resetSession(self):
        # Clear drawing
        self.canvas.delete("all")

        self.current_num_of_features = 0
        self.each_num_of_features = []
        self.num_of_drawings = 0
        self.labels = list()
        self.magnitudes = list()
        self.cartesian_cords = list()
        self.drawing_ovals = [[]]
        self.drawing_labels = list()
        self.drawing_magnitudes = list()
        self.drawing_cartesian_cords = [[]]

        # Update success status and options menu
        self.updateOptionsMenu(self.existing_labels, self.existing_label_text, self.labels, "No Labels")
        self.classify_results_text.set("Session Reset.")

    def get_window_entry(self, entry, window):
        self.session_name = entry
        window.destroy()

    def save(self, data, name):
        file_name = './saved/' + name + ".pickle"
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)  # Write to binary file
        f.close()
        self.zip.write(file_name, name + ".pickle")  # Add binary file to zip
        os.remove(file_name)  # Remove binary file

    def load(self, name):
        file_name = name + ".pickle"
        f = self.zip.open(file_name, "r")
        return pickle.load(f)

    def updateOptionsMenu(self, menu, menu_text, new_options, msg):
        # Update exising labels dropdown menu
        menu["menu"].delete(0, "end")
        for option in new_options:
            menu["menu"].add_command(label=option, command=tk._setit(menu_text, option))
        menu_text.set(msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.master.title("Drawing Classification")
    app.master.maxsize(500, 700)
    app.mainloop()
