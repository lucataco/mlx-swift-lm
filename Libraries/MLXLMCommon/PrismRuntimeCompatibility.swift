// Copyright © 2026 Apple Inc.
// Compatibility with PrismML-Eng/mlx-swift prism (pre-maskFill API).
import MLX

extension MLXArray {
    /// Match upstream's finite, dtype-preserving attention mask sentinel.
    public static func maskFill(for dtype: DType) -> MLXArray {
        let maximum: Double
        switch dtype {
        case .float16: maximum = 65_504
        case .bfloat16: maximum = Double(Float(bitPattern: 0x7F7F_0000))
        case .float32, .complex64: maximum = Double(Float.greatestFiniteMagnitude)
        case .float64: maximum = Double.greatestFiniteMagnitude
        default: preconditionFailure("maskFill requires a floating-point dtype")
        }
        return -MLXArray(maximum, dtype: dtype)
    }
}
