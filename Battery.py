"""
this scipt contains two LIB power battery models
an Equivalent circuit model
an electro-thermal-aging model
"""
import math
import pickle
from scipy.interpolate import interp1d

# LFP battery pack model based on Rint model 2015- Tang et al. EMS for HEV Including Battery Life Optimization
class CellModel1:
    def __init__(self):
        self.timestep = 1
        self.batt_maxpower = 40000  # W [40 kW]
        
        # Rint 모델 파라미터
        self.Cn = 6.5    # Ah 240V 기준으로 1.56kWh capacity는 6.5Ah로 계산
        self.soc_min = 0.01
        self.soc_max = 1.0
        # Capacity loss model constants -> 2015-Suri et al. 논문 기반 식에서 2.5Ah cell의 파라미터를 
        # 31Ah cell로 스케일링한 2025 Chang et al. 논문을 기반으로 6.5Ah cell에 맞게 조정
        self._loss_soc_split = 0.45         # SOC 기준
        self._loss_c1_low = 2129.4          # SOC <= 0.45 구간의 c1 상수
        self._loss_c2_low = 6991.7          # SOC <= 0.45 구간의 c2 상수
        self._loss_c1_high = 2093.4         # SOC > 0.45 구간의 c1 상수
        self._loss_c2_high = 5249.2         # SOC > 0.45 구간의 c2 상수
        self._loss_z = 0.57                 # Throughput 지수 (2025-Chang et al. 논문에서 LFP 셀에 맞게 조정)
        self._loss_t_env = 20.0             # 배터리 평균 온도 (섭씨, 2025-Chang et al. 논문에서 LFP 셀의 평균 운용 온도로 설정)
        # exponential values in Capacity loss calculation 
        self._loss_inv_rt = 1.0 / (8.314 * (self._loss_t_env + 273.15))
        self._loss_mAh_to_Ah = 1e-3         # mAh 단위를 Ah로 변환하기 위한 상수
        
        # OCV 맵핑 함수 2025 Chang et al. 논문 기반으로 배터리 팩의 SoC-OCV 데이터로 보간 함수 생성
        data_dir = "/home/chahyunjoon/dev/FCEV-EMS/project-data/Battery_data/"
        self.ocv_func = pickle.load(open(data_dir+'ocv_new.pkl', 'rb'))
        # R_in 맵핑 함수 (pack scale)
        self.r0_func = pickle.load(open(data_dir+'r_in.pkl', 'rb'))
    
    def run_battery(self, P_batt, paras_list):
        # 발열 관련 State(Tep_c, Tep_s, Tep_a) 모두 제거
        # paras_list = [SOC, SOH, Voc]
        SOC = paras_list[0]
        SOH = paras_list[1]
        Voc = paras_list[2]
        
        # 배터리 파워 제한
        if P_batt > self.batt_maxpower:
            P_batt = self.batt_maxpower
        if P_batt < -self.batt_maxpower:
            P_batt = -self.batt_maxpower
            
        # Rint 모델 전류 계산
        r0 = float(self.r0_func(SOC))
        delta = Voc**2 - 4 * r0 * P_batt
        if delta < 0: 
            I_batt = Voc / (2 * r0)
        else:
            I_batt = (Voc - math.sqrt(delta)) / (2 * r0)     
        
        # 1. SOC Update
        soc_deriv = self.timestep * (I_batt / 3600 / self.Cn)
        SOC_new = SOC - soc_deriv
        
        fail = False
        if SOC_new >= 1.0:
            SOC_new = 1.0  
            fail = True
        elif SOC_new <= 0.01:
            SOC_new = 0.01 
            fail = True

        #pack 단위 OCV, power 계산
        Voc_new = self.ocv_func(SOC_new)
        Vt_new = Voc_new - r0 * I_batt
        power_out = Vt_new * I_batt

        # 2. Aging Model (Bc 대신 Capacity Loss 적용)
        # 이번 스텝(1초) 동안 흐른 전하량 (Ah)
        current_step_Ah = (abs(I_batt) * self.timestep) / 3600.0
        
        Ic_rate = abs(I_batt) / self.Cn
        if SOC <= self._loss_soc_split:
            c1 = self._loss_c1_low
            c2 = self._loss_c2_low
        else:
            c1 = self._loss_c1_high
            c2 = self._loss_c2_high
        # Capacity Loss 계산 (2025-Chang et al. 논문 기반 공식 사용, 단위 변환 포함)
        q_loss_mAh = (c1 * SOC + c2) * math.exp((-31700.0 + 163.3 * Ic_rate) * self._loss_inv_rt) * (current_step_Ah ** self._loss_z)
        dsoh = (q_loss_mAh * self._loss_mAh_to_Ah) / self.Cn
        
        SOH_new = SOH - dsoh

        # 아웃풋 정보 정리
        out_info = {'SOC': SOC_new, 'SOH': SOH_new, 'soc_deriv': soc_deriv,
                    'pack_OCV': Voc_new, 'pack_Vt': Vt_new,
                    'I': I_batt, 'pack_power_out': power_out/1000,
                    'I_c': Ic_rate, 'P_batt_req': P_batt/1000,
                    'tep_a': self._loss_t_env, 'dsoh': dsoh}
                    
        # State 업데이트 (온도 관련 변수 제거)
        paras_list_new = [SOC_new, SOH_new, Voc_new]
        
        return paras_list_new, dsoh, I_batt, fail, out_info
