#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <chrono>       // 【新增】包含 chrono 头文件用于计时
#include <iomanip>      // 【新增】用于格式化输出时间
#include <Eigen/Dense> 
#include "SGP4.h" 

using namespace std;
using namespace Eigen;
using namespace SGP4Funcs; 

struct TLEData {
    string name;
    string line1;
    string line2;
};

class TLEReader {
public:
    static vector<TLEData> readAll(const string& filename) {
        vector<TLEData> database;
        ifstream file(filename);
        string name, l1, l2;
        
        while (getline(file, name)) {
            if (name.length() < 5) continue; 
            
            if (getline(file, l1) && getline(file, l2)) {
                database.push_back({name, l1, l2});
            }
        }
        return database;
    }
};

class CollisionAlgorithms {
public:
    static Matrix3d computeRotationMatrix(const Vector3d& r, const Vector3d& v) {
        Vector3d uR = r.normalized();
        Vector3d uN = (r.cross(v)).normalized();
        Vector3d uT = uN.cross(uR);
        Matrix3d M;
        M.col(0) = uR; M.col(1) = uT; M.col(2) = uN;
        return M;
    }

    static double calculateChanPc(const Vector3d& dr, const Vector3d& dv, 
                                 const Matrix3d& combinedCov, double radiusSum) {
        Vector3d ez = dv.normalized(); 
        Vector3d ref(1, 0, 0);
        if (abs(ez.dot(ref)) > 0.9) ref = Vector3d(0, 1, 0);
        Vector3d ex = ez.cross(ref).normalized();
        Vector3d ey = ez.cross(ex).normalized();

        Matrix3d M_enc; 
        M_enc.row(0) = ex; M_enc.row(1) = ey; M_enc.row(2) = ez;

        Vector3d r_proj = M_enc * dr;
        Matrix3d cov_3d_proj = M_enc * combinedCov * M_enc.transpose();
        Matrix2d C; 
        C << cov_3d_proj(0, 0), cov_3d_proj(0, 1),
             cov_3d_proj(1, 0), cov_3d_proj(1, 1);

        SelfAdjointEigenSolver<Matrix2d> eigensolver(C);
        Vector2d eigenvalues = eigensolver.eigenvalues();
        double sx2 = eigenvalues(0); double sy2 = eigenvalues(1);

        Vector2d pos_rot = eigensolver.eigenvectors().transpose() * Vector2d(r_proj(0), r_proj(1));
        double u = (pos_rot(0) * pos_rot(0) / sx2) + (pos_rot(1) * pos_rot(1) / sy2);
        double v = (radiusSum * radiusSum) / (sqrt(sx2 * sy2)); 
        return exp(-u / 2.0) * (1.0 - exp(-v / 2.0));
    }

    static double calculatePathSplittingPc(elsetrec& rec1, elsetrec& rec2, 
                                           double startJD, double endJD,
                                           const Matrix3d& combinedCov, 
                                           double radiusSum) {
        const double PI_C = 3.14159265358979323846;
        double stepSec = 30.0; 
        double stepJD = stepSec / 86400.0; 
        double pcTotal = 0.0;
        
        Matrix3d invC = combinedCov.inverse();
        double denom = sqrt(pow(2.0 * PI_C, 3) * combinedCov.determinant());
        double crossSectionArea = PI_C * radiusSum * radiusSum;

        for (double currentJD = startJD; currentJD <= endJD; currentJD += stepJD) {
            double ro1[3], vo1[3], ro2[3], vo2[3];
            sgp4(rec1, (currentJD - rec1.jdsatepoch) * 1440.0, ro1, vo1);
            sgp4(rec2, (currentJD - rec2.jdsatepoch) * 1440.0, ro2, vo2);
            
            Vector3d dr(ro2[0]-ro1[0], ro2[1]-ro1[1], ro2[2]-ro1[2]);
            Vector3d dv(vo2[0]-vo1[0], vo2[1]-vo1[1], vo2[2]-vo1[2]);
            
            double mahalanobisSq = dr.transpose() * invC * dr;
            if (mahalanobisSq > 25.0) continue; 
            pcTotal += (exp(-0.5 * mahalanobisSq) / denom) * (crossSectionArea * dv.norm() * stepSec);
        }
        return pcTotal;
    }
};

class ConjunctionFilter {
private:
    double distThreshold = 30.0;     
    double vRelThreshold = 0.1;      
    double radiusSum = 0.01;         
   
    bool findTCA(elsetrec& rec1, elsetrec& rec2, double startJD, double endJD, 
                 double& outTcaJD, double& outMinDist, 
                 Vector3d& outPos1, Vector3d& outPos2, 
                 Vector3d& outVel1, Vector3d& outVel2) {
        
        double stepCoarseJD = 60.0 / 86400.0; 
        double stepFineJD = 1.0 / 86400.0;    
        
        double minCoarseDist = 1e9;
        double minCoarseJD = startJD;

        for (double jd = startJD; jd <= endJD; jd += stepCoarseJD) {
            double ro1[3], vo1[3], ro2[3], vo2[3];
            sgp4(rec1, (jd - rec1.jdsatepoch) * 1440.0, ro1, vo1);
            sgp4(rec2, (jd - rec2.jdsatepoch) * 1440.0, ro2, vo2);
            
            Vector3d p1(ro1[0], ro1[1], ro1[2]);
            Vector3d p2(ro2[0], ro2[1], ro2[2]);
            double dist = (p1 - p2).norm();
            
            if (dist < minCoarseDist) {
                minCoarseDist = dist;
                minCoarseJD = jd;
            }
        }

        if (minCoarseDist > distThreshold * 2.0) {
            return false; 
        }

        double fineStart = std::max(startJD, minCoarseJD - stepCoarseJD);
        double fineEnd = std::min(endJD, minCoarseJD + stepCoarseJD);
        
        outMinDist = 1e9;
        
        for (double jd = fineStart; jd <= fineEnd; jd += stepFineJD) {
            double ro1[3], vo1[3], ro2[3], vo2[3];
            sgp4(rec1, (jd - rec1.jdsatepoch) * 1440.0, ro1, vo1);
            sgp4(rec2, (jd - rec2.jdsatepoch) * 1440.0, ro2, vo2);
            
            Vector3d p1(ro1[0], ro1[1], ro1[2]);
            Vector3d p2(ro2[0], ro2[1], ro2[2]);
            double dist = (p1 - p2).norm();
            
            if (dist < outMinDist) {
                outMinDist = dist;
                outTcaJD = jd;
                outPos1 = p1; outPos2 = p2;
                outVel1 = Vector3d(vo1[0], vo1[1], vo1[2]);
                outVel2 = Vector3d(vo2[0], vo2[1], vo2[2]);
            }
        }
        
        return true; 
    }

public:
    void processAllOnAll(const vector<TLEData>& catalog, double startJD, double endJD) {
        for (int i = 0; i < (int)catalog.size(); ++i) {
            for (int j = i + 1; j < (int)catalog.size(); ++j) {
                comparePair(catalog[i], catalog[j], startJD, endJD);
            }
        }
    }

private:
    void comparePair(const TLEData& t1, const TLEData& t2, double startJD, double endJD) {
        elsetrec rec1, rec2;
        double st, sp, dmin;
        char l1_1[130], l1_2[130], l2_1[130], l2_2[130];

        strcpy(l1_1, t1.line1.c_str()); strcpy(l1_2, t1.line2.c_str());
        twoline2rv(l1_1, l1_2, 'c', 't', 'i', wgs72, st, sp, dmin, rec1);
        
        strcpy(l2_1, t2.line1.c_str()); strcpy(l2_2, t2.line2.c_str());
        twoline2rv(l2_1, l2_2, 'c', 't', 'i', wgs72, st, sp, dmin, rec2);

        double tcaJD = 0.0;
        double minDist = 0.0;
        Vector3d pos1, pos2, vel1, vel2;

        if (!findTCA(rec1, rec2, startJD, endJD, tcaJD, minDist, pos1, pos2, vel1, vel2)) {
            return; 
        }

        if (minDist < distThreshold) {
            double sR = 0.2, sT = 2.0, sN = 0.2; 
            Matrix3d P_RTN = Matrix3d::Zero();
            P_RTN(0,0) = sR*sR; P_RTN(1,1) = sT*sT; P_RTN(2,2) = sN*sN;
            Matrix3d M1 = CollisionAlgorithms::computeRotationMatrix(pos1, vel1);
            Matrix3d M2 = CollisionAlgorithms::computeRotationMatrix(pos2, vel2);
            Matrix3d combinedCov = (M1 * P_RTN * M1.transpose()) + (M2 * P_RTN * M2.transpose());

            Vector3d relVel = vel2 - vel1;
            
            double pc = (relVel.norm() > vRelThreshold) ? 
                CollisionAlgorithms::calculateChanPc(pos2 - pos1, relVel, combinedCov, radiusSum) :
                CollisionAlgorithms::calculatePathSplittingPc(rec1, rec2, tcaJD - 0.02, tcaJD + 0.02, combinedCov, radiusSum);
            
            printf("[%s] %-18s vs %-18s | TCA(JD): %.5f | Dist: %9.3f km | Pc: %.2e\n", 
                   (relVel.norm() > vRelThreshold ? "HIGH-V" : "LOW-V "), 
                   t1.name.c_str(), t2.name.c_str(), tcaJD, minDist, pc);
        }
    }
};

int main() {
    // 【新增】记录程序开始时间
    auto startTime = chrono::high_resolution_clock::now();

    cout << "===== SGP4 Collision Detection Start =====" << endl;

    auto catalog = TLEReader::readAll("TLE.txt");
    cout << "[Step 1] File loaded. Total Satellites: " << catalog.size() << endl;

    if (catalog.empty()) {
        cout << "[ERROR] Catalog is empty. Please check TLE.txt format." << endl;
        return -1;
    }
    
    double startJD = 2460986.5; 
    double endJD = startJD + 1.0; 
    
    cout << "[Step 2] Analyzing conjunctions from JD " << fixed << setprecision(5) << startJD << " to " << endJD << endl;
    cout << "----------------------------------------------------------------" << endl;

    ConjunctionFilter filter;
    filter.processAllOnAll(catalog, startJD, endJD);

    cout << "----------------------------------------------------------------" << endl;
    
    // 【新增】记录程序结束时间
    auto endTime = chrono::high_resolution_clock::now();
    
    // 【新增】计算差值并转换为秒 (double 类型，带小数)
    chrono::duration<double> duration = endTime - startTime;
    
    cout << "===== Process Finished Successfully =====" << endl;
    cout << "[Performance] Total Execution Time: " << fixed << setprecision(3) << duration.count() << " seconds." << endl;

    return 0;
}