namespace TeamPitagorasProyecto.Services
{

    public record InverseMethodResult(
        bool Success,
        string? ErrorMessage,
        double[] Solution,
        double Determinant,
        double[,] InverseMatrix,
        List<string> Steps
    );

    public class MetodoInversaServices
    {
        private const double Epsilon = 1e-12;

        public InverseMethodResult Solve(double[,] matrixA, double[] vectorB)
        {
            int n = vectorB.Length;

            if (matrixA.GetLength(0) != n || matrixA.GetLength(1) != n || n < 2)
            {
                return new InverseMethodResult(
                    false,
                    "La matriz A debe ser cuadrada y coincidir con el tamano del vector b.",
                    Array.Empty<double>(),
                    0,
                    new double[0, 0],
                    new List<string>()
                );
            }

            var steps = new List<string>();
            double det = CalculateDeterminant((double[,])matrixA.Clone(), n);
            steps.Add($"Determinante de A: det(A) = {det:G6}");

            if (Math.Abs(det) < Epsilon)
            {
                return new InverseMethodResult(
                    false,
                    "El sistema no tiene solucion unica porque el determinante es cero (matriz singular).",
                    Array.Empty<double>(),
                    det,
                    new double[0, 0],
                    steps
                );
            }

            double[,] inverse = InvertMatrix(matrixA, n, steps);
            double[] solution = MultiplyMatrixVector(inverse, vectorB, n, steps);

            return new InverseMethodResult(
                true,
                null,
                solution,
                det,
                inverse,
                steps
            );
        }

        private double CalculateDeterminant(double[,] m, int n)
        {
            double det = 1;
            int sign = 1;

            for (int col = 0; col < n; col++)
            {
                int pivotRow = col;
                double maxVal = Math.Abs(m[col, col]);

                for (int row = col + 1; row < n; row++)
                {
                    double val = Math.Abs(m[row, col]);
                    if (val > maxVal)
                    {
                        maxVal = val;
                        pivotRow = row;
                    }
                }

                if (maxVal < Epsilon)
                {
                    return 0;
                }

                if (pivotRow != col)
                {
                    for (int k = col; k < n; k++)
                    {
                        (m[col, k], m[pivotRow, k]) = (m[pivotRow, k], m[col, k]);
                    }
                    sign = -sign;
                }

                double pivot = m[col, col];
                det *= pivot;

                for (int row = col + 1; row < n; row++)
                {
                    double factor = m[row, col] / pivot;
                    for (int k = col + 1; k < n; k++)
                    {
                        m[row, k] -= factor * m[col, k];
                    }
                }
            }

            return det * sign;
        }

        private double[,] InvertMatrix(double[,] a, int n, List<string> steps)
        {
            var aug = new double[n, 2 * n];

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    aug[i, j] = a[i, j];
                }
                aug[i, n + i] = 1.0;
            }

            steps.Add("Se construye la matriz aumentada [A | I] y se aplica Gauss-Jordan.");

            for (int col = 0; col < n; col++)
            {
                int pivotRow = col;
                double maxVal = Math.Abs(aug[col, col]);

                for (int row = col + 1; row < n; row++)
                {
                    double val = Math.Abs(aug[row, col]);
                    if (val > maxVal)
                    {
                        maxVal = val;
                        pivotRow = row;
                    }
                }

                if (pivotRow != col)
                {
                    for (int k = 0; k < 2 * n; k++)
                    {
                        (aug[col, k], aug[pivotRow, k]) = (aug[pivotRow, k], aug[col, k]);
                    }
                    steps.Add($"Intercambio de fila {col + 1} con fila {pivotRow + 1}.");
                }

                double pivot = aug[col, col];
                for (int k = 0; k < 2 * n; k++)
                {
                    aug[col, k] /= pivot;
                }
                steps.Add($"Normalizacion de fila {col + 1} dividiendo entre {pivot:G6}.");

                for (int row = 0; row < n; row++)
                {
                    if (row == col) continue;
                    double factor = aug[row, col];
                    for (int k = 0; k < 2 * n; k++)
                    {
                        aug[row, k] -= factor * aug[col, k];
                    }
                }
            }

            var inv = new double[n, n];
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    inv[i, j] = aug[i, n + j];
                }
            }

            steps.Add($"Matriz inversa A^-1:\n{FormatMatrix(inv, n)}");
            return inv;
        }

        private double[] MultiplyMatrixVector(double[,] inv, double[] b, int n, List<string> steps)
        {
            var x = new double[n];
            steps.Add("Calculo de X = A^-1 * b:");

            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                var terms = new List<string>();
                for (int j = 0; j < n; j++)
                {
                    sum += inv[i, j] * b[j];
                    terms.Add($"({inv[i, j]:G6})*({b[j]:G6})");
                }
                x[i] = Math.Abs(sum) < Epsilon ? 0 : sum;
                steps.Add($"x{i + 1} = {string.Join(" + ", terms)} = {x[i]:G6}");
            }

            return x;
        }

        private string FormatMatrix(double[,] m, int n)
        {
            var rows = new List<string>();
            for (int i = 0; i < n; i++)
            {
                var cols = new List<string>();
                for (int j = 0; j < n; j++)
                {
                    double val = Math.Abs(m[i, j]) < Epsilon ? 0 : m[i, j];
                    cols.Add($"{val,10:G6}");
                }
                rows.Add($"[ {string.Join("  ", cols)} ]");
            }
            return string.Join("\n", rows);
        }
    }
    
}
