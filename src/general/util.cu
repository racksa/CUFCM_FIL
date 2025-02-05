// util.cpp

#include "util.hpp"

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

std::unordered_map<std::string, std::unordered_map<std::string, std::string>> parseINI(const std::string& filename) {
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> data;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening the file: " << filename << std::endl;
        return data;
    }

    std::string current_section;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key, value;

        if (line.empty() || line[0] == ';' || line[0] == '#') {
            // Skip empty lines and comments
            continue;
        } else if (line[0] == '[' && line[line.length() - 1] == ']') {
            // Section line (e.g., [SectionName])
            current_section = line.substr(1, line.length() - 2);
        } else if (std::getline(iss, key, '=') && std::getline(iss, value)) {
            // Key-value pair line
            data[current_section][key] = value;
        }
    }

    return data;
}

std::string data_from_ini(std::string filename, std::string section, std::string variable){
    // std::string filename = "input/globals.ini";
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> iniData = parseINI(filename);

    // Access and use the parameters
    if (iniData.find(section) != iniData.end()) {
        std::cout << section << " " << variable << ": " << iniData[section][variable] << std::endl;
        return iniData[section][variable];        
    } else {
        std::cout << "Section " << section << " not found in the INI file." << std::endl;
        return "";
    }
}



CartesianCoordinates spherical_to_cartesian(double r, double theta, double phi) {
    CartesianCoordinates cartesian;
    cartesian.x = r * sin(theta) * cos(phi);
    cartesian.y = r * sin(theta) * sin(phi);
    cartesian.z = r * cos(theta);
    return cartesian;
}


CartesianCoordinates spherical_to_cartesian_field(double ur, double utheta, double uphi, 
                                          double theta, double phi) {
    double sin_theta = sin(theta);
    double cos_theta = cos(theta);
    double sin_phi = sin(phi);
    double cos_phi = cos(phi);

    CartesianCoordinates cartesian;

    cartesian.x = ur * sin_theta * cos_phi + utheta * cos_theta * cos_phi - uphi * sin_phi;
    cartesian.y = ur * sin_theta * sin_phi + utheta * cos_theta * sin_phi + uphi * cos_phi;
    cartesian.z = ur * cos_theta - utheta * sin_theta;

    return cartesian;
}


// Legendre Polynomial (same as before)
double legendreP(int N, double x) {
    if (N == 0) return 1.0;
    if (N == 1) return x;

    double Pn_minus_2 = 1.0; // P₀(x)
    double Pn_minus_1 = x;   // P₁(x)
    double Pn = 0.0;

    for (int k = 2; k <= N; ++k) {
        Pn = ((2 * k - 1) * x * Pn_minus_1 - (k - 1) * Pn_minus_2) / k;
        Pn_minus_2 = Pn_minus_1;
        Pn_minus_1 = Pn;
    }
    return Pn;
}

double derivative(int N, double x, double h = 1e-5) {
    return (legendreP(N, x + h) - legendreP(N, x - h)) / (2 * h);
}

// ---------- Associated Legendre Function (P_N^1(x)) ----------
double associated_legendre(int N, double x) {
    return -sqrt(1 - x * x) * derivative(N, x);
}

double legendreV(int N, double x) {
    if (N == 0) return 0.0; // To avoid division by zero
    return -2.0 / (N * (N + 1)) * associated_legendre(N, x);
}