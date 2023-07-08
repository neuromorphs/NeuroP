import sys
import random
import matplotlib
matplotlib.use('Qt5Agg')

from PySide6 import QtCore, QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from mpl_toolkits.basemap import Basemap


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        
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
        self.solution_canvas = MplCanvas(self, height=8, width=8, dpi=100)
        
        # TODO: make reasonable plot here
        m=Basemap(llcrnrlon=-100, llcrnrlat=-58,urcrnrlon=-30,urcrnrlat=15)
        m.drawcoastlines(color='black', linewidth=2, ax = self.solution_canvas.axes)
        
        # create the canvas for plotting the internal network state
        self.state_canvas = MplCanvas(self, dpi=100)
        
        # create the canvas for plotting the energy of the current solution
        self.energy_canvas = MplCanvas(self, dpi=100)
        
        plots_layout.addWidget(self.state_canvas)
        plots_layout.addWidget(self.energy_canvas)
        column_layout.addWidget(self.solution_canvas)
        column_layout.addLayout(plots_layout)
        
        # set the column layout for the windows
        widget = QtWidgets.QWidget()
        widget.setLayout(column_layout)
        self.setCentralWidget(widget)
        
        # Set up events
        self.chat_input.installEventFilter(self)

        n_data = 50
        self.xdata = list(range(n_data))
        self.ydata = [random.randint(0, 10) for i in range(n_data)]

        # We need to store a reference to the plotted line
        # somewhere, so we can apply the new data to it.
        self._plot_ref = None
        self.update_plot()

        self.show()

        # Setup a timer to trigger the redraw by calling update_plot.
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.KeyPress and source is self.chat_input):
            key = event.key()
            if key == QtCore.Qt.Key_Return or key == QtCore.Qt.Key_Enter:
                self.update_chat_input()
                return True
            
        return QtWidgets.QWidget.eventFilter(self, source, event)

    def update_chat_input(self):
        self.chat_output_text = self.chat_input.toPlainText()
        self.chat_output.setStyleSheet("background-color: #CCCCCC;")
        
        # TODO: Replace by actual API query to ChatGPT
        QtCore.QTimer.singleShot(1000,self.update_chat_output)
        
    def update_chat_output(self):
        self.chat_output.setStyleSheet("background-color: #333333;")
        self.chat_output.setPlainText(self.chat_output_text)


    def update_plot(self):
        # Drop off the first y element, append a new one.
        self.ydata = self.ydata[1:] + [random.randint(0, 10)]

        # Note: we no longer need to clear the axis.
        if self._plot_ref is None:
            # First time we have no plot reference, so do a normal plot.
            # .plot returns a list of line <reference>s, as we're
            # only getting one we can take the first element.
            plot_refs = self.energy_canvas.axes.plot(self.xdata, self.ydata, 'r')
            self._plot_ref = plot_refs[0]
        else:
            # We have a reference, we can use it to update the data for that line.
            self._plot_ref.set_ydata(self.ydata)

        # Trigger the canvas to update and redraw.
        self.energy_canvas.draw()

app = QtWidgets.QApplication(sys.argv)
w = MainWindow()
app.exec()