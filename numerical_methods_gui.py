
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import sympy as sp


# ================= FUNCTION HANDLING =================

x_sym = sp.Symbol('x')


def get_function(expression):
    expr = sp.sympify(expression)
    f = sp.lambdify(x_sym, expr, 'numpy')
    return f


def get_points(text):
    xs, ys = [], []

    for line in text.strip().splitlines():
        values = line.replace(',', ' ').split()

        if len(values) != 2:
            raise ValueError("Each line must contain x and y.")

        xs.append(float(values[0]))
        ys.append(float(values[1]))

    if len(xs) < 2:
        raise ValueError("Enter at least two data points.")

    if len(set(xs)) != len(xs):
        raise ValueError("X values must be different.")

    return np.array(xs), np.array(ys)


def metrics(ys, predictions):
    residuals = ys - predictions
    sse = np.sum(residuals ** 2)
    ss_tot = np.sum((ys - np.mean(ys)) ** 2)

    r2 = 1 - sse / ss_tot if ss_tot != 0 else float('nan')
    rmse = np.sqrt(sse / len(ys))

    return residuals, sse, r2, rmse


# ================= MAIN APPLICATION =================

class NumericalMethodsGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Numerical Methods Project")
        self.root.geometry("1050x750")
        self.root.minsize(850, 650)

        self.root.configure(bg="#F3F4F8")

        self.fields = {}

        self.create_styles()
        self.create_header()
        self.create_layout()

        self.change_method()

    # ================= STYLING =================

    def create_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TCombobox",
            padding=8,
            font=("Segoe UI", 11)
        )

        style.configure(
            "TEntry",
            padding=8,
            font=("Segoe UI", 11)
        )

    # ================= HEADER =================

    def create_header(self):
        header = tk.Frame(
            self.root,
            bg="#242B49",
            height=110
        )
        header.pack(fill="x")
        header.pack_propagate(False)

        tk.Label(
            header,
            text="NUMERICAL METHODS",
            font=("Segoe UI", 24, "bold"),
            bg="#242B49",
            fg="white"
        ).pack(pady=(18, 2))

        tk.Label(
            header,
            text="Numerical Analysis & Computational Mathematics",
            font=("Segoe UI", 10),
            bg="#242B49",
            fg="#BFC8E8"
        ).pack()

    # ================= MAIN LAYOUT =================

    def create_layout(self):

        main = tk.Frame(self.root, bg="#F3F4F8")
        main.pack(fill="both", expand=True, padx=25, pady=20)

        # Method selection

        tk.Label(
            main,
            text="Select Numerical Method",
            font=("Segoe UI", 12, "bold"),
            bg="#F3F4F8",
            fg="#242B49"
        ).pack(anchor="w")

        self.method = ttk.Combobox(
            main,
            values=[
                "Lagrange Interpolation",
                "Non-Linear Regression",
                "Linear Regression",
                "Bisection Method",
                "Secant Method"
            ],
            state="readonly",
            font=("Segoe UI", 11)
        )

        self.method.pack(fill="x", pady=(7, 18))
        self.method.current(0)
        self.method.bind("<<ComboboxSelected>>", self.change_method)

        # Input and output area

        content = tk.Frame(main, bg="#F3F4F8")
        content.pack(fill="both", expand=True)

        self.input_frame = tk.LabelFrame(
            content,
            text=" Input Parameters ",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#242B49",
            padx=15,
            pady=15
        )

        self.input_frame.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 10)
        )

        self.output_frame = tk.LabelFrame(
            content,
            text=" Results ",
            font=("Segoe UI", 11, "bold"),
            bg="white",
            fg="#242B49",
            padx=12,
            pady=12
        )

        self.output_frame.pack(
            side="right",
            fill="both",
            expand=True,
            padx=(10, 0)
        )

        # Output text

        self.output = tk.Text(
            self.output_frame,
            font=("Consolas", 10),
            bg="#F8F9FC",
            fg="#242B49",
            wrap="none",
            relief="flat",
            padx=10,
            pady=10
        )

        self.output.pack(fill="both", expand=True)

        # Buttons

        buttons = tk.Frame(main, bg="#F3F4F8")
        buttons.pack(fill="x", pady=(15, 0))

        tk.Button(
            buttons,
            text="Calculate",
            command=self.calculate,
            font=("Segoe UI", 11, "bold"),
            bg="#5967D8",
            fg="white",
            activebackground="#424FBD",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            padx=30,
            pady=10
        ).pack(side="left")

        tk.Button(
            buttons,
            text="Clear",
            command=self.clear,
            font=("Segoe UI", 11, "bold"),
            bg="#E5E7EF",
            fg="#242B49",
            relief="flat",
            cursor="hand2",
            padx=30,
            pady=10
        ).pack(side="left", padx=10)

    # ================= INPUT FIELDS =================

    def add_entry(self, label, default=""):

        row = tk.Frame(self.input_frame, bg="white")
        row.pack(fill="x", pady=7)

        tk.Label(
            row,
            text=label,
            font=("Segoe UI", 10),
            bg="white",
            fg="#444B65"
        ).pack(anchor="w")

        entry = ttk.Entry(row)
        entry.pack(fill="x", pady=(5, 0))

        if default != "":
            entry.insert(0, str(default))

        self.fields[label] = entry

    def add_text(self, label, default=""):

        tk.Label(
            self.input_frame,
            text=label,
            font=("Segoe UI", 10),
            bg="white",
            fg="#444B65"
        ).pack(anchor="w", pady=(10, 5))

        text = tk.Text(
            self.input_frame,
            height=7,
            font=("Consolas", 10),
            bg="#F8F9FC",
            relief="solid",
            bd=1
        )

        text.pack(fill="x")

        if default:
            text.insert("1.0", default)

        self.fields[label] = text

    def change_method(self, event=None):

        for widget in self.input_frame.winfo_children():
            widget.destroy()

        self.fields = {}

        selected = self.method.get()

        if selected == "Bisection Method":

            self.add_entry("Function f(x)", "x**2 - 4")
            self.add_entry("a", "1")
            self.add_entry("b", "3")
            self.add_entry("Tolerance", "0.0001")
            self.add_entry("Max Iterations", "100")

        elif selected == "Secant Method":

            self.add_entry("Function f(x)", "x**2 - 4")
            self.add_entry("x0", "1")
            self.add_entry("x1", "3")
            self.add_entry("Tolerance", "0.0001")
            self.add_entry("Max Iterations", "100")

        elif selected == "Linear Regression":

            self.add_text(
                "Data Points (x, y) - One pair per line",
                "1, 3\n2, 5\n3, 7\n4, 9"
            )
            self.add_entry("Predict at x (optional)", "")

        elif selected == "Non-Linear Regression":

            self.add_entry(
                "Model (1=Exponential, 2=Power, 3=Quadratic)",
                "1"
            )

            self.add_text(
                "Data Points (x, y) - One pair per line",
                "0, 2\n1, 4\n2, 8\n3, 16"
            )

        elif selected == "Lagrange Interpolation":

            self.add_text(
                "Data Points (x, y) - One pair per line",
                "1, 1\n2, 4\n3, 9"
            )

            self.add_entry("Interpolate at x", "2.5")

        self.output.delete("1.0", "end")

        self.output.insert(
            "end",
            "Select your parameters and click Calculate."
        )

    # ================= OUTPUT HELPERS =================

    def write(self, text):
        self.output.insert("end", str(text) + "\n")

    def show_table(self, headers, rows):

        widths = [
            max(len(str(h)), max(
                (len(str(row[i])) for row in rows),
                default=0
            ))
            for i, h in enumerate(headers)
        ]

        header = " | ".join(
            str(h).ljust(widths[i])
            for i, h in enumerate(headers)
        )

        self.write(header)
        self.write("-" * len(header))

        for row in rows:
            self.write(
                " | ".join(
                    str(v).ljust(widths[i])
                    for i, v in enumerate(row)
                )
            )

    def get_value(self, label):
        return self.fields[label].get().strip()

    # ================= CALCULATION =================

    def calculate(self):

        self.output.delete("1.0", "end")

        try:

            selected = self.method.get()

            self.write("NUMERICAL METHODS PROJECT")
            self.write("=" * 45)
            self.write(f"Method: {selected}\n")

            if selected == "Bisection Method":
                self.bisection()

            elif selected == "Secant Method":
                self.secant()

            elif selected == "Linear Regression":
                self.linear_regression()

            elif selected == "Non-Linear Regression":
                self.nonlinear_regression()

            elif selected == "Lagrange Interpolation":
                self.lagrange()

        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    # ================= BISECTION =================

    def bisection(self):

        f = get_function(self.get_value("Function f(x)"))

        a = float(self.get_value("a"))
        b = float(self.get_value("b"))
        tol = float(self.get_value("Tolerance"))
        max_iter = int(self.get_value("Max Iterations"))

        if tol <= 0 or max_iter <= 0:
            raise ValueError("Tolerance and iterations must be positive.")

        fa, fb = float(f(a)), float(f(b))

        if fa == 0:
            self.write(f"Root = {a}")
            return

        if fb == 0:
            self.write(f"Root = {b}")
            return

        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs.")

        rows = []
        old_c = None

        for i in range(1, max_iter + 1):

            c = (a + b) / 2
            fc = float(f(c))

            error = abs(c - old_c) if old_c is not None else 0

            rows.append([
                i,
                round(a, 6),
                round(b, 6),
                round(c, 6),
                round(fc, 6),
                round(error, 6)
            ])

            if abs(fc) < tol or (old_c is not None and error < tol):
                break

            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc

            old_c = c

        self.show_table(
            ["Iter", "a", "b", "c", "f(c)", "Error"],
            rows
        )

        self.write(f"\nApproximate Root = {c}")
        self.write(f"f(root) = {fc}")
        self.write(f"Iterations = {i}")

        if i == max_iter and abs(fc) >= tol and (
            len(rows) < 2 or rows[-1][-1] >= tol
        ):
            self.write("Maximum iterations reached.")

    # ================= SECANT =================

    def secant(self):

        f = get_function(self.get_value("Function f(x)"))

        x0 = float(self.get_value("x0"))
        x1 = float(self.get_value("x1"))
        tol = float(self.get_value("Tolerance"))
        max_iter = int(self.get_value("Max Iterations"))

        if tol <= 0 or max_iter <= 0:
            raise ValueError("Tolerance and iterations must be positive.")

        rows = []

        for i in range(1, max_iter + 1):

            f0 = float(f(x0))
            f1 = float(f(x1))

            if abs(f1 - f0) < 1e-15:
                raise ValueError("Division by zero. Change initial guesses.")

            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

            error = abs(x2 - x1)

            rows.append([
                i,
                round(x0, 6),
                round(x1, 6),
                round(x2, 6),
                round(float(f(x2)), 6),
                round(error, 6)
            ])

            if error < tol or abs(float(f(x2))) < tol:
                break

            x0, x1 = x1, x2

        self.show_table(
            ["Iter", "x0", "x1", "x2", "f(x2)", "Error"],
            rows
        )

        self.write(f"\nApproximate Root = {x2}")
        self.write(f"f(root) = {float(f(x2))}")
        self.write(f"Iterations = {i}")

        if i == max_iter and error >= tol:
            self.write("Maximum iterations reached.")

    # ================= LINEAR REGRESSION =================

    def linear_regression(self):

        xs, ys = get_points(
            self.get_value("Data Points (x, y) - One pair per line")
        )

        n = len(xs)

        sum_x = np.sum(xs)
        sum_y = np.sum(ys)
        sum_xy = np.sum(xs * ys)
        sum_x2 = np.sum(xs ** 2)

        denominator = n * sum_x2 - sum_x ** 2

        if abs(denominator) < 1e-15:
            raise ValueError("Cannot calculate slope: all x values are equal.")

        a1 = (n * sum_xy - sum_x * sum_y) / denominator
        a0 = (sum_y - a1 * sum_x) / n

        predictions = a0 + a1 * xs

        residuals, sse, r2, rmse = metrics(ys, predictions)

        self.write(f"Sum x   = {sum_x}")
        self.write(f"Sum y   = {sum_y}")
        self.write(f"Sum xy  = {sum_xy}")
        self.write(f"Sum x^2 = {sum_x2}")

        self.write(f"\nSlope (a1) = {a1:.6f}")
        self.write(f"Intercept (a0) = {a0:.6f}")
        self.write(f"\nEquation: y = {a0:.6f} + {a1:.6f}x")

        rows = [
            [round(x, 4), round(y, 4),
             round(p, 4), round(e, 4)]
            for x, y, p, e in zip(xs, ys, predictions, residuals)
        ]

        self.show_table(
            ["x", "Actual", "Predicted", "Error"],
            rows
        )

        self.write(f"\nR² = {r2:.6f}")
        self.write(f"RMSE = {rmse:.6f}")
        self.write(f"SSE = {sse:.6f}")

        new_x = self.get_value("Predict at x (optional)")

        if new_x:
            new_x = float(new_x)
            prediction = a0 + a1 * new_x
            self.write(f"\nPredicted y({new_x}) = {prediction:.6f}")

    # ================= NON-LINEAR REGRESSION =================

    def nonlinear_regression(self):

        choice = self.get_value(
            "Model (1=Exponential, 2=Power, 3=Quadratic)"
        )

        xs, ys = get_points(
            self.get_value("Data Points (x, y) - One pair per line")
        )

        if choice == "1":

            if np.any(ys <= 0):
                raise ValueError("Exponential model requires y > 0.")

            Y = np.log(ys)

            A = np.vstack([np.ones_like(xs), xs]).T

            coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)

            ln_a, b = coeffs
            a = np.exp(ln_a)

            predictions = a * np.exp(b * xs)

            self.write(f"Model: y = {a:.6f} * exp({b:.6f}x)")

        elif choice == "2":

            if np.any(xs <= 0) or np.any(ys <= 0):
                raise ValueError("Power model requires x > 0 and y > 0.")

            X = np.log(xs)
            Y = np.log(ys)

            A = np.vstack([np.ones_like(X), X]).T

            coeffs, *_ = np.linalg.lstsq(A, Y, rcond=None)

            ln_a, b = coeffs
            a = np.exp(ln_a)

            predictions = a * (xs ** b)

            self.write(f"Model: y = {a:.6f} * x^({b:.6f})")

        elif choice == "3":

            A = np.vstack([
                np.ones_like(xs),
                xs,
                xs ** 2
            ]).T

            coeffs, *_ = np.linalg.lstsq(A, ys, rcond=None)

            a0, a1, a2 = coeffs

            predictions = a0 + a1 * xs + a2 * xs ** 2

            self.write(
                f"Model: y = {a0:.6f} + "
                f"{a1:.6f}x + {a2:.6f}x²"
            )

        else:
            raise ValueError("Choose model 1, 2, or 3.")

        residuals, sse, r2, rmse = metrics(ys, predictions)

        rows = [
            [round(x, 4), round(y, 4),
             round(p, 4), round(e, 4)]
            for x, y, p, e in zip(xs, ys, predictions, residuals)
        ]

        self.show_table(
            ["x", "Actual", "Predicted", "Error"],
            rows
        )

        self.write(f"\nR² = {r2:.6f}")
        self.write(f"RMSE = {rmse:.6f}")
        self.write(f"SSE = {sse:.6f}")

    # ================= LAGRANGE =================

    def lagrange(self):

        xs, ys = get_points(
            self.get_value("Data Points (x, y) - One pair per line")
        )

        x_val = float(self.get_value("Interpolate at x"))

        result = 0
        rows = []

        for i in range(len(xs)):

            Li = 1

            for j in range(len(xs)):

                if i != j:
                    Li *= (x_val - xs[j]) / (xs[i] - xs[j])

            contribution = ys[i] * Li
            result += contribution

            rows.append([
                i,
                round(xs[i], 4),
                round(ys[i], 4),
                round(Li, 6),
                round(contribution, 6)
            ])

        self.show_table(
            ["i", "x_i", "y_i", "L_i", "Contribution"],
            rows
        )

        self.write(f"\nInterpolated Value = {result:.8f}")

    # ================= CLEAR =================

    def clear(self):

        self.change_method()


# ================= RUN APPLICATION =================

if __name__ == "__main__":

    root = tk.Tk()

    app = NumericalMethodsGUI(root)

    root.mainloop()
