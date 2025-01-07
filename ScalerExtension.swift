
import Foundation

extension Array where Element == Float {
    func scaled(withParameters params: ScalerParameters) -> [Float] {
        return self.enumerated().map { index, value in
            let x = value
            let min = params.dataMin[index]
            let range = params.dataRange[index]
            return range != 0 ? (x - min) / range : 0
        }
    }
}

struct ScalerParameters: Codable {
    let dataMin: [Float]
    let dataRange: [Float]
    
    static let `default`: ScalerParameters = {
        guard let url = Bundle.main.url(forResource: "scaler_params", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let params = try? JSONDecoder().decode(ScalerParameters.self, from: data) else {
            fatalError("Failed to load scaler parameters")
        }
        return params
    }()
}
