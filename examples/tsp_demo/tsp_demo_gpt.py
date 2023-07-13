import sys
import json
import random
import matplotlib
import openai
import numpy as np

matplotlib.use("Qt5Agg")
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtCore import QThreadPool, QRunnable, QObject, Signal, Slot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.basemap import Basemap  # https://github.com/matplotlib/basemap
import requests
import time

from solve_tsp import get_solution_tsp
from helpers import *

openai.api_key = "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"


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.thread_manager = QThreadPool()

        # Define your conversation history
        self.conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

        self.setStyleSheet(
            """
        QPlainTextEdit { background-color: #333333; color: #EEEEEE; }
        """
        )

        # main columns
        column_layout = QtWidgets.QHBoxLayout()

        # rows for chat prompt and reply
        chat_layout = QtWidgets.QVBoxLayout()

        # create the input field for the chat
        self.chat_input = QtWidgets.QPlainTextEdit(self)
        self.chat_input.setFixedWidth(400)

        # create the output field for the chat
        self.chat_output = QtWidgets.QPlainTextEdit(self)
        self.chat_output.setReadOnly(True)

        # add chat widgets
        chat_layout.addWidget(self.chat_input)
        chat_layout.addWidget(self.chat_output)
        column_layout.addLayout(chat_layout)

        # create the rows for the plots
        plots_layout = QtWidgets.QVBoxLayout()

        # create the canvas for plotting the problem solution
        #########################################################################
        self.solution_canvas = MplCanvas(self, height=800, width=600, dpi=100)

        # TODO: make reasonable plot here
        # self.map=Basemap(projection='cyl', llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90)

        # self.map.drawcoastlines(color='black', linewidth=2, ax = self.solution_canvas.axes)
        # self.map.drawcountries(color='black', linewidth=2, ax = self.solution_canvas.axes)

        # create the canvas for plotting the internal network state
        self.state_canvas = MplCanvas(self, dpi=100)

        # create the canvas for plotting the energy of the current solution
        self.energy_canvas = MplCanvas(self, dpi=100)
        ########################################################################
        plots_layout.addWidget(self.state_canvas)
        plots_layout.addWidget(self.energy_canvas)
        ##################################################################
        column_layout.addWidget(self.solution_canvas)
        ##################################################################
        column_layout.addLayout(plots_layout)

        # set the column layout for the windows
        widget = QtWidgets.QWidget()
        widget.setLayout(column_layout)
        self.setCentralWidget(widget)

        # Set up events
        self.chat_input.installEventFilter(self)

        # n_data = 50
        # self.xdata = list(range(n_data))
        # self.ydata = [random.randint(0, 10) for i in range(n_data)]

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self._plot_ref = None

        # self.update_plot()

        self.samples_buffer = []

        self.data_exists = False
        self.plot_done = False

        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(10)
        # self.timer.timeout.connect(self.update_plot)
        self.timer.timeout.connect(self.update_map)
        self.timer.start()
        question_text = ""  # "Describe the combinatorial opt problem:"
        self.chat_output.appendPlainText(question_text)

    def connect_coordinates(self, coord1, coord2, map, ax=None):
        x1, y1 = coord1
        x2, y2 = coord2
        map.drawgreatcircle(x1, y1, x2, y2, linewidth=2, color="r", ax=ax)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.KeyPress and source is self.chat_input:
            key = event.key()
            if key == QtCore.Qt.Key_Return or key == QtCore.Qt.Key_Enter:
                self.update_chat_input()
                return True

        return QtWidgets.QWidget.eventFilter(self, source, event)

    def get_coordinates(self, city_names):
        ans = []
        for city_name in city_names:
            url = f"https://geocode.maps.co/search?q={city_name}"
            response = requests.get(url).json()
            ans.append((float(response[0]["lat"]), float(response[0]["lon"])))
        return ans

    def plot_solutions(self, sample):
        N = len(self.distanceMatrix)
        m, schedule = parse_op_vec_tsp(sample, N)
        distance = get_distance(self.distanceMatrix, schedule, N)
        distance_km = int(round(distance / 1000, 0))

        destinations_scheduled = np.array(self.destination_lists)[schedule]
        coordinates_scheduled = np.array(self.coordinate_lists)[schedule]

        for i in range(0, len(coordinates_scheduled)):
            dest1 = destinations_scheduled[i - 1]
            dest2 = destinations_scheduled[i]
            coord1 = coordinates_scheduled[i - 1][::-1]
            coord2 = coordinates_scheduled[i][::-1]

            self.connect_coordinates(
                coord1, coord2, self.map, ax=self.solution_canvas.axes
            )

        self.solution_canvas.axes.set_title(f"Distance: {distance_km} km")
        self.solution_canvas.draw()

    def draw_map(self):
        lats, longs, labels = self.lats, self.longs, self.destination_lists

        border = 10
        self.solution_canvas.axes.clear()
        self.map = Basemap(
            projection="cyl",
            llcrnrlon=min(longs) - border,
            llcrnrlat=min(lats) - border,
            urcrnrlon=max(longs) + border,
            urcrnrlat=max(lats) + border,
        )
        self.map.drawcoastlines(color="gray", linewidth=2, ax=self.solution_canvas.axes)
        self.map.fillcontinents(
            color="lightgreen",
            lake_color="cornflowerblue",
            ax=self.solution_canvas.axes,
        )
        self.map.drawcountries(color="gray", linewidth=2, ax=self.solution_canvas.axes)

        for lat, long, label in zip(lats, longs, labels):
            # Plot the cities on the map
            self.map.scatter(
                long, lat, marker="o", color="k", ax=self.solution_canvas.axes
            )
            # Annotate the cities' names
            self.solution_canvas.axes.text(long+0.5, lat+0.01, label, color='k', fontsize=7)
        
        self.solution_canvas.draw()

    def tsp_solver(self, destination_lists, coordinate_lists):

        # get the coordinates of the destinations
        is_realcity = not len(coordinate_lists)
        if is_realcity:
            coordinate_lists = self.get_coordinates(destination_lists)
        problem_info = {
            "destination_lists": destination_lists,
            "coordinate_lists": coordinate_lists,
        }

        print(destination_lists)

        self.destination_lists = destination_lists
        self.coordinate_lists = coordinate_lists

        print(coordinate_lists)
        self.lats = [coordinate[0] for coordinate in coordinate_lists]
        self.longs = [coordinate[1] for coordinate in coordinate_lists]

        self.draw_map()

        self.thread_manager.start(self.tsp_solver_actual) # Calling the actual solver in a separate thread

    def tsp_solver_actual(self):
        print("-----FLAG: TSP SOLVER-----")

        SOLVE = True

        self.distanceMatrix = get_distance_matrix(self.coordinate_lists)
        # np.save("distanceMatrix_TSP_10.npy", distanceMatrix)
        self.N = len(self.distanceMatrix)

        if not SOLVE:
            sorted_samples_by_energy = np.load("sorted_samples_by_energy.npy")
            m, schedule = parse_op_vec_tsp(sorted_samples_by_energy[0], self.N)
            distance = get_distance(self.distanceMatrix, schedule, self.N)

        else:
            # Index of the city to start with
            start_city = 0
            # Do we want to enforce start city: 0 no, 1 yes
            enforce_start_city = 1
            schedule, distance, sorted_samples_by_energy = get_solution_tsp(
                self.destination_lists, self.distanceMatrix, start_city, enforce_start_city
            )

            m, schedule_ = parse_op_vec_tsp(sorted_samples_by_energy[0], self.N)

        self.sorted_samples_by_energy = np.array(sorted_samples_by_energy)
        self.samples_buffer = self.sorted_samples_by_energy.tolist()
        self.data_exists = True
        self.plot_done = False
        distance_km = int(round(distance / 1000, 0))

        destinations_scheduled = np.array(self.destination_lists)[schedule]

        ans = {"best path": destinations_scheduled.tolist()}

        print("Done.")
        self.update_chat_output_gpt(json.dumps(ans))

        # return json.dumps(ans)

    def maxcut_solver(self, adj_matrix):
        print("-----FLAG: MAXCUT SOLVER-----")
        problem_info = {
            "adj_matrix": adj_matrix,
        }
        maxcut = 0  # maxcut_gtnn(np.array(json.loads(problem_info["adj_matrix"])))
        ans = {"maxcuts": maxcut}
        return json.dumps(ans)

    def update_chat_input(self):
        # self.chat_output_text = self.chat_input.toPlainText()
        # self.chat_output.setStyleSheet("background-color: #CCCCCC;")

        user_input = self.chat_input.toPlainText()
        if user_input == None:
            user_input = "In the context of a MaxCut problem, calculate the number of maxcuts for the following undirected graph: 'Node 1 is connected to node 2 and node 3, node 2 is connected to node 3.'"

        self.messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that helps to solve combinatorial problems.\
                    Your tasks include: 1. extract useful information from the plain-text description of the problem\
                    2. call the corresponding solver functions to solve the problem with the extracted information.\
                    You might receive one of the two types of problem: MaxCut, TSP (travelling sales person).\
                    For the MaxCut problems, you are expected to convert the user-specified graph into adjacency matrix and call the maxcut solver\
                    For the TSP problems, there are two possible scenarios:\
                    when the list of destinations are real-world cities, such as Beijing, New York, Tokyo, the list of destinations will be whatever the user has specified, and list of coordinates will be none.\
                    when the list of destinations are symbolic, such as A, B, C, the list of destinations and list of coordinates are whatever the user has specified. Whenever you provide coordinates, make sure to provide them in floats and not in strings.",
                "role": "user",
                "content": user_input,
            }
        ]
        functions = [
            {
                "name": "maxcut_solver",
                "description": "Calculate the maxcut for a maxcut problem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "adj_matrix": {
                            "type": "string",
                            "description": "The adjacency matrix to be further processed, e.g. M = [[0, 1], [0, 0]]",
                        },
                    },
                    "required": ["adj_matrix"],
                },
            },
            {
                "name": "tsp_solver",
                "description": "Calculate the path vector for a TSP problem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "destination_lists": {
                            "type": "string",
                            "description": "The lists of destination that the traverling sales person want to travel with the starting location as the first destination in the lists, e.g. D = ['New York, NY', 'Los Angeles, CA'] or D = ['A', 'B', 'C', 'D']",
                        },
                        "coordinate_lists": {
                            "type": "string",
                            "description": " The coordinates of the destinations. If the destinations are real-world cities, provide the real-world (longitude, latitude) for the city, e.g. C = [(40.7128, -74.0060), (34.0522, -118.2437)] or if destinations are symbolic and user specifies the coordinates of the destinations, this parameter will be the user -specified coordinates, e.g.  C = [(10, 20), (20, 10)]",
                        },
                    },
                    "required": ["destination_lists", "coordinate_lists"],
                },
            },
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages,
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit
        )
        print(response)
        # print(response)
        response_message = response["choices"][0]["message"]
        if response_message.get("function_call"):
            print(response_message)
            print("-----FLAG: INVOKED SOLVERS-----")
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "maxcut_solver": self.maxcut_solver,
                "tsp_solver": self.tsp_solver,
            }  # only one function in this example, but you can have multiple
            self.function_name = response_message["function_call"]["name"]
            fuction_to_call = available_functions[self.function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            if self.function_name == "maxcut_solver":
                function_response = fuction_to_call(
                    adj_matrix=function_args.get("adj_matrix"),
                )
                pass
            elif self.function_name == "tsp_solver":
                destination_lists = function_args.get("destination_lists")
                coordinate_lists = function_args.get("coordinate_lists")
                function_response = fuction_to_call(
                    destination_lists=destination_lists,
                    coordinate_lists=coordinate_lists,
                )
                pass

    def update_chat_output_gpt(self, function_response):
        print("-----FLAG: UPDATE CHAT OUTPUT-----")
        print(self.messages)

        self.messages.append(
            {
                "role": "function",
                "name": self.function_name,
                "content": function_response,
            })

        second_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=self.messages,
        )                          

        self.update_chat_output(second_response.choices[0].message.content)

    def update_chat_output(self, second_response):

        self.chat_output.clear()
        self.chat_output_text = second_response
        self.chat_output.setStyleSheet("background-color: #333333;")
        # self.chat_output.setPlainText(self.chat_output_text)

        self.chat_output.appendPlainText(second_response)

        self.chat_output.verticalScrollBar().setValue(
            self.chat_output.verticalScrollBar().maximum()
        )

    def update_map(self):
        # We want to update the map here

        if not self.samples_buffer:
            pass

        else:
            if self.data_exists:
                self.draw_map()
            sample = self.samples_buffer.pop(-1)
            self.plot_solutions(sample)

            if not self.samples_buffer:
                self.data_exists = False
                self.plot_done = True

    # def update_plot(self):
    #     # Drop off the first y element, append a new one.
    #     self.ydata = self.ydata[1:] + [random.randint(0, 10)]

    #     # Note: we no longer need to clear the axis.
    #     if self._plot_ref is None:
    #         # First time we have no plot reference, so do a normal plot.
    #         # .plot returns a list of line <reference>s, as we're
    #         # only getting one we can take the first element.
    #         plot_refs = self.energy_canvas.axes.plot(self.xdata, self.ydata, 'r')
    #         self._plot_ref = plot_refs[0]
    #     else:
    #         # We have a reference, we can use it to update the data for that line.
    #         self._plot_ref.set_ydata(self.ydata)

    #     # Trigger the canvas to update and redraw.
    #     self.energy_canvas.draw()


app = QtWidgets.QApplication(sys.argv)
w = MainWindow()

app.exec()
