import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QFrame, QPushButton)
from PyQt5.QtCore import Qt, pyqtSignal


class FilteredComboBox(QWidget):
    """
    Custom dropdown widget with filtering capability
    """
    # Signal emitted when an item is selected
    itemSelected = pyqtSignal(str)

    def __init__(self, items=None, parent=None):
        super().__init__(parent)

        self.items = items or []
        self.filtered_items = self.items.copy()
        self.is_dropdown_visible = False

        self.setup_ui()
        self.setup_connections()
        self.update_list()

    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Main display field (shows selected item)
        self.display_field = QLineEdit()
        self.display_field.setPlaceholderText("Select an item...")
        self.display_field.setReadOnly(False)  # Allow typing for filtering
        layout.addWidget(self.display_field)

        # Dropdown container
        self.dropdown_container = QFrame(self)
        self.dropdown_container.setFrameShape(QFrame.StyledPanel)
        self.dropdown_container.setFrameShadow(QFrame.Raised)
        self.dropdown_container.setVisible(False)

        dropdown_layout = QVBoxLayout(self.dropdown_container)
        dropdown_layout.setContentsMargins(0, 0, 0, 0)
        dropdown_layout.setSpacing(0)

        # List widget for items
        self.list_widget = QListWidget()
        dropdown_layout.addWidget(self.list_widget)

        layout.addWidget(self.dropdown_container)

        # Set minimum width
        self.setMinimumWidth(200)

    def setup_connections(self):
        # Toggle dropdown when clicking on the display field
        self.display_field.mousePressEvent = self.toggle_dropdown

        # Item selection and filtering
        self.list_widget.itemClicked.connect(self.on_item_clicked)
        self.display_field.textChanged.connect(self.filter_items)

    def toggle_dropdown(self, event=None):
        # Toggle dropdown visibility
        self.is_dropdown_visible = not self.is_dropdown_visible
        self.dropdown_container.setVisible(self.is_dropdown_visible)

        # Update dropdown position and size
        if self.is_dropdown_visible:
            self.dropdown_container.setFixedWidth(self.display_field.width())
            max_height = min(200, self.list_widget.sizeHintForRow(0) * (self.list_widget.count() + 2))
            self.dropdown_container.setFixedHeight(max_height)

            # Focus on the line edit for immediate filtering
            self.display_field.setFocus()

    def on_item_clicked(self, item):
        # Update display field with selected text
        self.display_field.setText(item.text())
        self.display_field.setCursorPosition(0)

        # Hide dropdown
        self.is_dropdown_visible = False
        self.dropdown_container.setVisible(False)

        # Emit signal with selected item
        self.itemSelected.emit(item.text())

    def filter_items(self, text):
        if not self.is_dropdown_visible:
            # If we're typing but dropdown is hidden, show it
            self.is_dropdown_visible = True
            self.dropdown_container.setVisible(True)

        # Filter items based on current text
        self.filtered_items = [item for item in self.items
                               if text.lower() in item.lower()]
        self.update_list()

    def update_list(self):
        # Clear and repopulate list widget
        self.list_widget.clear()
        for item in self.filtered_items:
            self.list_widget.addItem(QListWidgetItem(item))

    def setItems(self, items):
        # Update items list
        self.items = items
        self.filtered_items = items.copy()
        self.update_list()

    def getCurrentText(self):
        # Get current selected text
        return self.display_field.text()

    def setCurrentText(self, text):
        # Set current text
        self.display_field.setText(text)

    # Override focusOutEvent to close dropdown when focus is lost
    def focusOutEvent(self, event):
        # Small delay to allow item click to register before hiding dropdown
        # Without this, dropdown might close before the click is registered
        QApplication.processEvents()
        if not self.list_widget.underMouse():
            self.is_dropdown_visible = False
            self.dropdown_container.setVisible(False)
        super().focusOutEvent(event)


class DemoWindow(QMainWindow):
    """Demo application to show the filtered combo box in action"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Filtered Dropdown Demo")
        self.setGeometry(300, 300, 400, 200)

        # Sample data
        countries = [
            "United States", "Canada", "Mexico", "Brazil", "Argentina",
            "United Kingdom", "France", "Germany", "Italy", "Spain",
            "China", "Japan", "India", "Australia", "Russia"
        ]

        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Label
        layout.addWidget(QLabel("Select a country:"))

        # Filtered dropdown
        self.filtered_combo = FilteredComboBox(countries)
        self.filtered_combo.itemSelected.connect(self.on_selection_changed)
        layout.addWidget(self.filtered_combo)

        # Selected value display
        self.selection_label = QLabel("Selected: None")
        layout.addWidget(self.selection_label)

        # Add some space
        layout.addStretch()

    def on_selection_changed(self, text):
        self.selection_label.setText(f"Selected: {text}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Apply some basic styling
    app.setStyleSheet("""
        QLineEdit {
            padding: 6px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        QListWidget {
            border: none;
        }
        QListWidget::item {
            padding: 6px;
        }
        QListWidget::item:hover {
            background-color: #e6e6e6;
        }
        QListWidget::item:selected {
            background-color: #0078d7;
            color: white;
        }
    """)

    window = DemoWindow()
    window.show()
    sys.exit(app.exec_())