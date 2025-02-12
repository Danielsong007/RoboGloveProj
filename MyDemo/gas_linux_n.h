#ifndef GAS_LINUX_N_H
#define GAS_LINUX_N_H
#define MAX_MACRO_CHAR_LENGTH (128)

//函数执行返回值
#define GA_COM_SUCCESS			        (0)	//执行成功
#define GA_COM_ERR_EXEC_FAIL			(1)	//执行失败
#define GA_COM_ERR_LICENSE_WRONG		(2)	//license不支持
#define GA_COM_ERR_DATA_WORRY			(7)	//参数错误
#define GA_COM_ERR_SEND					(-1)//发送失败
#define GA_COM_ERR_CARD_OPEN_FAIL		(-6)//打开失败
#define GA_COM_ERR_TIME_OUT				(-7)//无响应
#define GA_COM_ERR_COM_OPEN_FAIL        (-8)//打开串口失败

//轴状态位定义
#define AXIS_STATUS_ESTOP               (0x00000001)	//急停
#define AXIS_STATUS_SV_ALARM            (0x00000002)	//驱动器报警标志（1-伺服有报警，0-伺服无报警）
#define AXIS_STATUS_POS_SOFT_LIMIT      (0x00000004)	//正软限位触发标志（规划位置大于正向软限位时置1）
#define AXIS_STATUS_NEG_SOFT_LIMIT      (0x00000008)	//负软位触发标志（规划位置小于负向软限位时置1）
#define	AXIS_STATUS_FOLLOW_ERR          (0x00000010)	//轴规划位置和实际位置的误差大于设定极限时置1┅
#define AXIS_STATUS_POS_HARD_LIMIT      (0x00000020)	//正硬限位触发标志（正限位开关电平状态为限位触发电平时置1）
#define AXIS_STATUS_NEG_HARD_LIMIT      (0x00000040)	//负硬限位触发标志（负限位开关电平状态为限位触发电平时置1）
#define AXIS_STATUS_IO_SMS_STOP         (0x00000080)	//IO平滑停止触发标志（正限位开关电平状态为限位触发电平时置1，规划位置大于正向软限位时置1）
#define AXIS_STATUS_IO_EMG_STOP         (0x00000100)	//IO紧急停止触发标志（负限位开关电平状态为限位触发电平时置1，规划位置小于负向软限位时置1）
#define AXIS_STATUS_ENABLE              (0x00000200)	//电机使能标志
#define	AXIS_STATUS_RUNNING             (0x00000400)	//规划运动标志，规划器运动时置1
#define AXIS_STATUS_ARRIVE              (0x00000800)	//电机到位（规划器静止，规划位置和实际位置的误差小于设定误差带，并在误差带内保持设定时间后，置起到位标志）
#define AXIS_STATUS_HOME_RUNNING        (0x00001000)	//正在回零
#define AXIS_STATUS_HOME_SUCESS	        (0x00002000)	//回零成功
#define AXIS_STATUS_HOME_SWITCH			(0x00004000)	//零位信号
#define AXIS_STATUS_INDEX				(0x00008000)    //z索引信号
#define AXIS_STATUS_GEAR_START  		(0x00010000)    //电子齿轮开始啮合
#define AXIS_STATUS_GEAR_FINISH         (0x00020000)    //电子齿轮完成啮合
#define AXIS_STATUS_HOME_FAIL  	        (0x00400000)	//回零失败
#define AXIS_STATUS_ECAT_HOME  	        (0x00800000)	//伺服零位
#define AXIS_STATUS_ECAT_PROBE	        (0x01000000)	//伺服探针

//坐标系状态位定义
#define	CRDSYS_STATUS_PROG_RUN						(0x00000001)	//启动中
#define	CRDSYS_STATUS_PROG_STOP						(0x00000002)	//平滑停止中
#define	CRDSYS_STATUS_PROG_ESTOP					(0x00000004)	//紧急停止中
#define	CRDSYS_STATUS_FIFO_FINISH_0					(0x00000010)	//板卡FIFO-0数据已执行完毕的状态位
#define	CRDSYS_STATUS_FIFO_FINISH_1					(0x00000020)	//板卡FIFO-1数据已执行完毕的状态位
#define	CRDSYS_STATUS_ALARM      					(0x00000040)	//坐标系有报警
#define	CRDSYS_STATUS_OFFLINE      					(0x00000080)	//EtherCAT离线或者异常

//输入IO类型宏定义
#define MC_LIMIT_POSITIVE               0
#define MC_LIMIT_NEGATIVE               1
#define MC_ALARM                        2
#define MC_HOME                         3
#define MC_GPI                          4
#define MC_ARRIVE                       5
#define MC_IP_SWITCH                    6
#define MC_MPG                          7

//输出IO类型宏定义
#define MC_ENABLE                       10
#define MC_CLEAR                        11
#define MC_GPO                          12


//高速捕获输入类型宏定义
#define CAPTURE_HOME                    1//HOME捕获
#define CAPTURE_INDEX                   2//INDEX捕获
#define CAPTURE_PROBE1                  3//探针捕获
#define CAPTURE_PROBE2                  4
#define CAPTURE_X0                      100
#define CAPTURE_X1                      101
#define CAPTURE_X2                      102
#define CAPTURE_X3                      103
#define CAPTURE_X4                      104
#define CAPTURE_X5                      105
#define CAPTURE_X6                      106
#define CAPTURE_X7                      107
#define CAPTURE_X8                      108
#define CAPTURE_X9                      109
#define CAPTURE_X10                     110
#define CAPTURE_X11                     111
#define CAPTURE_X12                     112
#define CAPTURE_X13                     113
#define CAPTURE_X14                     114
#define CAPTURE_X15                     115

#define CAPTURE_HOME_STOP               11//HOME捕获，并自动停止
#define CAPTURE_INDEX_STOP              12//INDEX捕获，并自动停止
#define CAPTURE_PROBE1_STOP             13//探针捕获，并自动停止
#define CAPTURE_PROBE2_STOP             14
#define CAPTURE_X0_STOP                 200
#define CAPTURE_X1_STOP                 201
#define CAPTURE_X2_STOP                 202
#define CAPTURE_X3_STOP                 203
#define CAPTURE_X4_STOP                 204
#define CAPTURE_X5_STOP                 205
#define CAPTURE_X6_STOP                 206
#define CAPTURE_X7_STOP                 207
#define CAPTURE_X8_STOP                 208
#define CAPTURE_X9_STOP                 209
#define CAPTURE_X10_STOP                210
#define CAPTURE_X11_STOP                211
#define CAPTURE_X12_STOP                212
#define CAPTURE_X13_STOP                213
#define CAPTURE_X14_STOP                214
#define CAPTURE_X15_STOP                215

//PT模式宏定义
#define PT_MODE_STATIC                  0
#define PT_MODE_DYNAMIC                 1

#define PT_SEGMENT_NORMAL               0
#define PT_SEGMENT_EVEN                 1
#define PT_SEGMENT_STOP                 2

#define GEAR_MASTER_ENCODER             1//电子齿轮，跟随编码器
#define GEAR_MASTER_PROFILE             2//电子齿轮，跟随规划值(脉冲个数)
#define GEAR_MASTER_AXIS                3//保留


//电子齿轮启动事件定义
#define GEAR_EVENT_IMMED                1//立即启动电子齿轮
#define GEAR_EVENT_BIG_EQU              2//主轴规划或者编码器位置大于等于指定数值时启动电子齿轮
#define GEAR_EVENT_SMALL_EQU            3//主轴规划或者编码器位置小于等于指定数值时启动电子齿轮
#define GEAR_EVENT_IO_ON                4//指定IO为ON时启动电子齿轮
#define GEAR_EVENT_IO_OFF               5//指定IO为OFF时启动电子齿轮

#define CAM_EVENT_IMMED                1
#define CAM_EVENT_BIG_EQU              2
#define CAM_EVENT_SMALL_EQU            3
#define CAM_EVENT_IO_ON                4
#define CAM_EVENT_IO_OFF               5

#define FROCAST_LEN (200)                //前瞻缓冲区深度

#define INTERPOLATION_AXIS_MAX          6
#define CRD_FIFO_MAX                    4096
#define CRD_MAX                         2


//点位模式参数结构体
typedef struct TrapPrm
{
	double acc;//加速度
	double dec;//减速度
	double velStart;//起始速度
	short  smoothTime;//平滑时间
}TTrapPrm;

//JOG模式参数结构体
typedef struct JogPrm
{
	double dAcc;//加速度
	double dDec;//减速度
	double dSmooth;//平滑时间
}TJogPrm;

//坐标系参数结构体
typedef struct _CrdPrm
{
    short dimension;                              // 坐标系维数
    short profile[8];                             // 关联profile和坐标轴(从1开始)
    double synVelMax;                             // 最大合成速度
    double synAccMax;                             // 最大合成加速度
    short evenTime;                               // 最小匀速时间
    short setOriginFlag;                          // 设置原点坐标值标志,0:默认当前规划位置为原点位置;1:用户指定原点位置
    int originPos[8];                            // 用户指定的原点位置
}TCrdPrm;

//坐标系参数结构体
typedef struct _CrdPrmEx
{
    short dimension;                              // 坐标系维数
    short profile[32];                             // 关联profile和坐标轴(从1开始)
    double synVelMax;                             // 最大合成速度
    double synAccMax;                             // 最大合成加速度
    short evenTime;                               // 最小匀速时间
    short setOriginFlag;                          // 设置原点坐标值标志,0:默认当前规划位置为原点位置;1:用户指定原点位置
    int originPos[32];                            // 用户指定的原点位置
}TCrdPrmEx;

//TXPDO和RXPDO参数
typedef struct _ECatPDOParm{
	unsigned char cTXPDOCount;//TXPDO条目数量
	unsigned char cRXPDOCount;//RXPDO条目数量

	unsigned int lTXPDOItem[10];//TXPDO条目
	unsigned int lRXPDOItem[10];//RXPDO条目
}TECatPDOParm;

//命令类型
enum _CMD_TYPE
{
	CMD_EMPTY=0,
	CMD_G00=1,		//快速定位
	CMD_G01,		//直线插补
	CMD_G02,		//顺圆弧插补
	CMD_G03,		//逆圆弧插补
	CMD_G04,		//延时,G04 P1000是暂停1秒(单位为ms),G04 X2.0是暂停2秒
	CMD_G05,		//设置自定义插补段段号
	CMD_G54,
	CMD_HELIX_G02,		//顺圆弧螺旋插补
	CMD_HELIX_G03,		//逆圆弧螺旋插补
	CMD_CRD_PRM,        //建立坐标系

	CMD_M00 = 11,  //暂停
	CMD_M30,        //结束
	CMD_M31,        //切换到XY1Z坐标系
	CMD_M32,        //切换到XY2Z坐标系
	CMD_M99,        //循环

	CMD_SET_IO = 101,     //设置IO
	CMD_WAIT_IO,           //等待IO
	CMD_BUFFER_MOVE_SET_POS,      
	CMD_BUFFER_MOVE_SET_VEL,      
	CMD_BUFFER_MOVE_SET_ACC,      
	CMD_BUFFER_GEAR,      //BUFFER_GEAR
	CMD_BUFFER_JOG,           //启动指定轴速度运动
	CMD_BUFFER_ZERO_POS,      //指定轴位置清零
	CMD_SET_CRD_ORG_POS,      //设置坐标系原点
	CMD_SET_M,
	CMD_WAIT_M,
	CMD_BUFFER_LASER_FOLLOW_RATIO,
	CMD_CHECK_VARIABLE_POINT,
	CMD_BUFFER_PWM,
	CMD_BUFFER_DA,
	CMD_BUFFER_JUMP,
	CMD_BUFFER_BACK_UP_CRD_ORG,                  //备份CRD_ORG
	CMD_BUFFER_RE_STORE_CRD_ORG,                 //恢复CRD_ORG
	CMD_BUFFER_ABS_DATA,
	CMD_BUFFER_REL_DATA,
	CMD_BUFFER_CMP_OUT,      //比较输出
	CMD_BUFFER_MULTI_POINT,      //多点
	CMD_BUFFER_MULTI_POINT_END,      //多点结束
};

//机器人类型
enum _ROBOT_TYPE
{
	ROBOT_NULL=0,

	ROBOT_DELTA = 40001,  //Delta机械手

	ROBOT_XYZA  = 50001,  //XYZ+A摇篮
	ROBOT_XYZB  = 50002,  //XYZ+B摇篮
	ROBOT_XYZC  = 50003,  //XYZ+C摇篮

	ROBOT_XYZTA  = 50004,  //XYZ+A摆头
	ROBOT_XYZTB  = 50005,  //XYZ+B摆头
	ROBOT_XYZTC  = 50006,  //XYZ+C摆头

	ROBOT_XYZAB = 50007,  //XYZ+AB摇篮
	ROBOT_XYZAC = 50008,  //XYZ+AC摇篮
	ROBOT_XYZBC = 50009,  //XYZ+BC摇篮

	ROBOT_XYZTATB = 50010,//XYZ+AB双摆头
	ROBOT_XYZTATC = 50011,//XYZ+AC双摆头
	ROBOT_XYZTBTC = 50012,//XYZ+BC双摆头
	
};

//M00(暂停)命令参数
struct _M00PARA{
	int segNum;
};

struct _G00_1_PRE_PARA{
	int lEndC;
	int lEnd[8];
};

//G00(快速定位)命令参数
struct _G00PARA{
	float synVel; //插补段合成速度
	float synAcc; //插补段合成加速度
    int lX;       //X轴到达位置绝对位置(单位：pluse)
    int lY;       //Y轴到达位置绝对位置(单位：pluse)
    int lZ;       //Z轴到达位置绝对位置(单位：pluse)
    int lA;       //A轴到达位置绝对位置(单位：pluse)
	unsigned char iDimension; //参与插补的轴数量
	unsigned char cFuncFlag; //0代表不走阵列，1代表走第一组阵列，2代表走第二组阵列.......
    int segNum;
    int lB;       //B轴到达位置绝对位置(单位：pluse)(放在这里兼容老版本，位置不能随便移动)
    int lDisMask; //屏蔽掩码，对应位为1代表该轴不运动
};

//多点命令参数
struct _MULTI_POINT_PARA{
	unsigned int nTotalPoint;
	unsigned short nStepPerPoint;
	int segNum;
};

//结束多点命令参数
struct _MULTI_POINT_END_PARA{
	int segNum;
};

//G01(直线插补)命令参数(任意2到3轴，上位机保证)
struct _G01PARA{
	float synVel;    //插补段合成速度
	float synAcc;    //插补段合成加速度
	float velEnd;   //插补段的终点速度
    int lX;       //X轴到达位置绝对位置(单位：pluse)
    int lY;       //Y轴到达位置绝对位置(单位：pluse)
    int lZ;       //Z轴到达位置绝对位置(单位：pluse)
    int lA;       //A轴到达位置绝对位置(单位：pluse)
    int segNum;
	unsigned char iDimension; //参与插补的轴数量
	unsigned char iPreciseStopFlag;   //精准定位标志位，如果为1，终点按照终点坐标来
    int lB;                //B轴到达位置绝对位置(单位：pluse)(放在这里兼容老版本，位置不能随便移动)
};

//G02_G03(顺圆弧插补)命令参数(任意2轴，上位机保证)
struct _G02_3PARA{
	float synVel;    //插补段合成速度
	float synAcc;    //插补段合成加速度
	float velEnd;   //插补段的终点速度
    int iPlaneSelect;       //平面选择0：XY平面 1：XZ平面 2：YZ平面
    int iEnd1;              //第一轴终点坐标（单位um）
    int iEnd2;              //第二轴终点坐标（单位um）
    int iI;                 //圆心坐标（单位um）(相对于起点)
    int iJ;                 //圆心坐标（单位um）(相对于起点)
    int segNum;
    unsigned char iPreciseStopFlag;   //精准定位标志位，如果为1，终点按照终点坐标来
};

struct _Elliptic_G02_3_PRE_PARA{
	int lR1;
	int lR2;
};

//G02_G03(椭圆插补)命令参数(任意2轴，上位机保证)
struct _Elliptic_G02_3PARA{
	float synVel;           //插补段合成速度,单位脉冲/毫秒
	float synAcc;           //插补段合成加速度，单位脉冲/毫秒^2
	float velEnd;           //插补段的终点速度,单位脉冲/毫秒
    int iPlaneSelect;       //平面选择0：XY平面 1：XZ平面 2：YZ平面
    int iEnd1;              //第一轴终点坐标（单位脉冲）
    int iEnd2;              //第二轴终点坐标（单位脉冲）
    int iI;                 //圆心坐标（单位脉冲）(相对于起点)
    int iJ;                 //圆心坐标（单位脉冲）(相对于起点)
	int segNum;            //用户自定义行号
	unsigned char iPreciseStopFlag;   //精准定位标志位，如果为1，终点按照终点坐标来
};

typedef struct _HelixTypeBit{
	  unsigned char ParaType:2;//参数类型选择，0，Z+K模式，1，Count+K模式
	  unsigned char PlaneSelect:3;//平面选择0：XY平面 1：XZ平面 2：YZ平面
	  unsigned char Reserve:3;
}THelixTypeBit;

struct _HELIX_G02_3_PRE_PARA{
    int lEndX;
    int lEndY;
    int lEndZ;
    int segNum;            //用户自定义行号
};

//HELIX_G02_G03(螺旋线插补)命令参数(任意2轴，上位机保证)
struct _HELIX_G02_3PARA{
	float synVel;           //插补段合成速度,单位脉冲/毫秒
	float synAcc;           //插补段合成加速度，单位脉冲/毫秒^2
	float velEnd;           //插补段的终点速度,单位脉冲/毫秒

	float CirlceCount;      //圈数
	float k;                //螺旋线的导程

    int iI;                 //圆心坐标（单位脉冲）(相对于起点)
    int iJ;                 //圆心坐标（单位脉冲）(相对于起点)
	int lEnd3;             //第三轴到达位置绝对位置(单位：pluse)
	int segNum;            //用户自定义行号

	THelixTypeBit HelixTypeBit;
	unsigned char cPreciseStopFlag;//精准定位标志位，如果为1，终点按照终点坐标来
};

//G04延时
struct _G04PARA{
unsigned int ulDelayTime;       //延时时间,单位MS
int segNum;
};

//G05设置用户自定义段号
struct _G05PARA{
int lUserSegNum;       //用户自定义段号
};

//BufferMove命令参数(最多支持8轴)
struct _BufferMoveGearPARA{
    int lAxis1Pos[8];         //轴目标位置，最大支持8轴。轴的加速度和速度采用点位运动速度和加速度。该轴必须处于点位模式且不是插补轴
    int lUserSegNum;          //用户自定义行号
	unsigned char cAxisMask;   //轴掩码，bit0代表轴1，bit1代表轴2，.......对应位为1代表该轴要bufferMove
	unsigned char cModalMask;  //轴掩码，bit0代表轴1，bit1代表轴2，.......对应位为1代表该轴为阻塞，该轴到位后才进入下一行
};

//BufferMove设置Vel和Acc命令参数(最多支持8轴)
struct _BufferMoveVelAccPARA{
	float dVelAcc[8];          //轴速度及加速度，最大支持8轴。
    int lUserSegNum;          //用户自定义行号
	unsigned char cAxisMask;   //轴掩码，bit0代表轴1，bit1代表轴2，.......对应位为1代表该轴要bufferMove
};

//SetIO设置物理IO
struct _SetIOPara{
	unsigned short nCarkIndex;  //板卡索引，0代表主卡，1代表扩展卡1，2代表扩展卡2......依次类推
	unsigned short nDoMask;
	unsigned short nDoValue;
    int lUserSegNum;
};

//SetIO设置物理IO
struct _SetIOReversePara{
	unsigned short nCarkIndex;  //板卡索引，0代表主卡，1代表扩展卡1，2代表扩展卡2......依次类推
	unsigned short nDoMask;
	unsigned short nDoValue;
	unsigned short nReverseTime;
    int lUserSegNum;
};

//WaitIO等待物理IO输入
struct _WaitIOPara{
	unsigned short nIOCardIndex;//卡索引，0代表主卡，1代表扩展卡1，依次类推
	unsigned short nIOPortIndex;//IO索引，0~15
	unsigned short nLevel;     //0低电平1高电平
	unsigned short nFilterTime;//滤波时间，单位ms
    unsigned int  lWaitTimeMS;//单位毫秒，0代表一直等待
    int lUserSegNum;
};

//SetM设置信号量
struct _SetMPara{
	int iMAddr;
	int iMValue;
	int iBlockFlag;//0不阻塞，1阻塞直到信号量消失
    int lUserSegNum;
};

//WaitM等待信号量
struct _WaitMPara{
	int iMAddr;
	int iMValue;//0低电平1高电平    
	int iWaitTimeMS;//单位毫秒，0代表一直等待
    int lUserSegNum;
};

//BufferJog参数结构体
struct _BufferJogPara{
	short nAxisNum;
	short nBlock;
	double dVel;
	double dAccDec;
    int lUserSegNum;
};

//BufferZeroPos参数结构体
struct _BufferZeroPosPara{
	short nAxisNum;
    int lUserSegNum;
};

//BufCrdPrm
struct _BufCrdPrm{
	short dimension;           //坐标系维数
	short profile[8];          //关联Profile和坐标轴(从1开始)
	double synVelMax;          //最大合成速度
	double synAccMax;          //最大合成加速度
	short evenTime;            //最小匀速时间
};

//设置坐标系原点
struct _SetCrdOrgPos{
    int lX;       //X轴到达位置绝对位置(单位：pluse)
    int lY;       //Y轴到达位置绝对位置(单位：pluse)
    int lZ;       //Z轴到达位置绝对位置(单位：pluse)
    int lA;       //A轴到达位置绝对位置(单位：pluse)
    int lB;       //B轴到达位置绝对位置(单位：pluse)
	int iFlag;     //0设置指定点为坐标系原点，1设置当前所在点为坐标系原点
};

//检查阵列完成事件
struct _CheckVariableEvent{
    int iArrayNum;//阵列点编号
    int segNum;
};

//跳转构体
struct _BufferJumpPar{
	char Type;                 //变量类型，0、IO,1、M变量，2、D变量
	int VariableAddr;          //变量地址
	char Condition;            //0等于，1大于，2小于
	int CmpValue;              //比较值
	int Offset;                //偏移
    int lUserSegNum;
};

//BufPWM
struct _BufPWM{
	short nPwmNum;
	double dFreq;
	double dDuty;
	int lUserSegNum;
};

//BufDA
struct _BufDA{
	short nDacNum;
	short nValue;
	int lUserSegNum;
};

//G代码参数
union _CMDPara{
	struct _M00PARA     M00PARA;
    struct _G00PARA     G00PARA;
    struct _G01PARA     G01PARA;
    struct _G02_3PARA   G02_3PARA;
	struct _HELIX_G02_3PARA     HELIX_G02_3PARA;
    struct _G04PARA     G04PARA;
    struct _G05PARA     G05PARA;
	struct _BufferMoveGearPARA  BufferMoveGearPARA;
	struct _BufferMoveVelAccPARA BufferMoveVelAccPARA;
	struct _SetIOPara   SetIOPara;
	struct _SetIOReversePara SetIOReversePara;
	struct _WaitIOPara  				WaitIOPara;
	struct _SetMPara    SetMPara;
	struct _WaitMPara  	WaitMPara;
	struct _BufferJogPara       BufferJogPara;
	struct _BufferZeroPosPara   BufferZeroPosPara;
	struct _BufCrdPrm      BufCrdPrm;
	struct _SetCrdOrgPos BufSetCrdOrgPos;
	struct _CheckVariableEvent CheckVariableEvent;
	struct _BufPWM      BufPWM;
	struct _BufDA       BufDA;
	struct _BufferJumpPar BufferJumpPar;
	struct _HELIX_G02_3_PRE_PARA HELIX_G02_3_PRE_PARA;
	struct _G00_1_PRE_PARA G00_1_PRE_PARA;
};

//每一行程序结构体
typedef struct _CrdData{
    unsigned char CMDType;              //指令类型，支持最多255种指令0：GOO 1：G01 2：G02 FF:文件结束
    union _CMDPara CMDPara;         //指令参数，不同命令对应不同参数
}TCrdData;

//前瞻参数结构体
typedef struct _LookAheadPrm
{
	int lookAheadNum;                               //前瞻段数
	double dSpeedMax[INTERPOLATION_AXIS_MAX];	    //各轴的最大速度(p/ms)
	double dAccMax[INTERPOLATION_AXIS_MAX];			//各轴的最大加速度
	double dMaxStepSpeed[INTERPOLATION_AXIS_MAX];   //各轴的最大速度变化量（相当于启动速度）
	double dScale[INTERPOLATION_AXIS_MAX];			//各轴的脉冲当量

	//这个指针变量一定要放到最后，因为指针变量再32位系统下长度是32，在64位系统下长度是64
	TCrdData * pLookAheadBuf;                       //前瞻缓冲区指针
}TLookAheadPrm;

//轴回零参数
typedef struct _AxisHomeParm{
	short		nHomeMode;					//回零方式：0--无 1--HOME回原点	2--HOME加Index回原点3----Z脉冲	
	short		nHomeDir;					//回零方向，1-正向回零，0-负向回零
    int        lOffset;                    //回零偏移，回到零位后再走一个Offset作为零位

	double		dHomeRapidVel;			    //回零快移速度，单位：Pluse/ms
	double		dHomeLocatVel;			    //回零定位速度，单位：Pluse/ms
	double		dHomeIndexVel;			    //回零寻找INDEX速度，单位：Pluse/ms
	double      dHomeAcc;                   //回零使用的加速度

	int ulHomeIndexDis;           //找Index最大距离
	int ulHomeBackDis;            //回零时，第一次碰到零位后的回退距离
	unsigned short nDelayTimeBeforeZero;    //位置清零前，延时时间，单位ms
	unsigned int ulHomeMaxDis;//回零最大寻找范围，单位脉冲
}TAxisHomePrm;

//系统状态结构体
typedef struct _AllSysStatusData
{
	double dAxisEncPos[9];//轴编码器位置，包含一个手轮
	double dAxisPrfPos[8];//轴规划位置
    unsigned int lAxisStatus[8];//轴状态
	short nADCValue[2];//ADC值
    int lUserSegNum[2];//两个坐标系的用户段号
    int lRemainderSegNum[2];//两个坐标系的剩余段号
	short nCrdRunStatus[2];//两个坐标系的坐标系状态
    int lCrdSpace[2];//两个坐标系的剩余空间
	double dCrdVel[2];//两个坐标系的速度
	double dCrdPos[2][5];//两个坐标系的坐标
    int lLimitPosRaw;//正硬限位
    int lLimitNegRaw;//负硬限位
    int lAlarmRaw;//报警输入
    int lHomeRaw;//零位输入
    int lMPG;//手轮信号
    int lGpiRaw[4];//通用IO输入（除主卡外，最大支持3个扩展模块）
}TAllSysStatusData;

//16轴以内系统状态结构体
typedef struct _AllSysStatusDataSX
{
	int lAxisEncPos[16];//轴编码器位置
	int lAxisPrfPos[16];//轴规划位置
	unsigned int lAxisStatus[16];//轴状态
	short nADCValue[2];//ADC值
	int lUserSegNum[2];//两个坐标系的用户段号
	short lRemainderSegNum[2];//两个坐标系的剩余段号
	short nCrdRunStatus[2];//两个坐标系的坐标系状态
	short lCrdSpace[2];//两个坐标系的剩余空间
	float dCrdVel[2];//两个坐标系的速度
	int lCrdPos[2][5];//两个坐标系的坐标
	short lLimitPosRaw;//正硬限位
	short lLimitNegRaw;//负硬限位
	short lAlarmRaw;//报警输入
	short lHomeRaw;//零位输入
	int lMPGEncPos;//手轮编码器
	int lMPG;//手轮IO信号
	int lGpiRaw[8];//通用IO输入（除主卡外，最大支持7个扩展模块）
	int lGpoRaw[8];//通用IO输出（除主卡外，最大支持7个扩展模块）
}TAllSysStatusDataSX;

//Delta参数
typedef struct _DeltaParm{ 
	int  lPlusePerCircle[3];   //关节电机每圈脉冲数，单位脉冲
	double dRotateAngle[3];     //关节平面相对XZ平面的旋转角度，单位角度
	double dDisFixPlatform[3];  //定平台中心点到连接点的长度，单位mm
	double dLengthArm1[3];      //活动关节1的臂长
	double dLengthArm2[3];      //活动关节2的臂长
	double dDisMovPlatform[3];	//动平台中心点到连接点的长度，单位mm
}DELTA_PARM;


//其他指令列表
int GA_Open(short iType=0,char* cName="COM1");
int GA_OpenByIP(char* cPCIP,char* cCardIP,unsigned int ulID,unsigned short nRetryTime);
int GA_Close(void);
int GA_SetCardNo(short iCardNum);
int GA_GetCardNo(short *pCardNum);
int GA_Reset();

int GA_GetVersion(char *pVersion);
int GA_SetPrfPos(short profile,int prfPos);
int GA_SynchAxisPos(int mask);
int GA_ZeroPos(short nAxisNum,short nCount=1);
int GA_SetAxisBand(short nAxisNum,int lBand,int lTime);
int GA_GetAxisBand(short nAxisNum,int *pBand,int *pTime);
int GA_SetBacklash(short nAxisNum,int lCompValue,double dCompChangeValue,int lCompDir);
int GA_GetBacklash(short nAxisNum,int *pCompValue,double *pCompChangeValue,int *pCompDir);
int GA_SendString(char* cString,int iLen,int iOpenFlag=0);
int GA_SetPCEthernetPort(unsigned short nPCEthernetPort);
int GA_SetBaudRate(int BaudRate);
int GA_FwUpdate(char *File,unsigned int ulFileLen,int *pProgress);
int GA_GetFPGAVersion(unsigned int* pVersion);
int GA_GetCpuUsage(float* pUsage);

//系统配置信息
int GA_HomeSns(unsigned short sense);
int GA_GetHomeSns(short *pSense);
int GA_AlarmOn(short nAxisNum);
int GA_AlarmOff(short nAxisNum);
int GA_GetAlarmOnOff(short nAxisNum,short *pAlarmOnOff);
int GA_AlarmSns(unsigned short nSense);
int GA_GetAlarmSns(short *pSense);
int GA_LmtsOn(short nAxisNum,short limitType=-1);
int GA_LmtsOff(short nAxisNum,short limitType=-1);
int GA_GetLmtsOnOff(short nAxisNum,short *pPosLmtsOnOff, short *pNegLmtsOnOff);
int GA_LmtSns(unsigned short nSense);
int GA_LmtSnsEX(unsigned int lSense);
int GA_GetLmtSns(unsigned int *pSense);
int GA_SetLmtSnsSingle(short nAxisNum,short nPosSns,short nNegSns);
int GA_GetLmtSnsSingle(short nAxisNum,short* nPosSns,short* nNegSns);
int GA_ProfileScale(short nAxisNum,short alpha,short beta);
int GA_EncScale(short nAxisNum,short alpha,short beta);
int GA_StepDir(short step);
int GA_StepPulse(short step);
int GA_GetStep(short nAxisNum,short *pStep);
int GA_StepSns(unsigned short sense);
int GA_GetStepSns(short *pSense);
int GA_EncSns(unsigned short sense);
int GA_GetEncSns(short *pSense);
int GA_EncOn(short nEncoderNum);
int GA_EncOff(short nEncoderNum);
int GA_GetEncOnOff(short nAxisNum,short *pEncOnOff);
int GA_SetPosErr(short nAxisNum,int lError);
int GA_GetPosErr(short nAxisNum,int *pError);
int GA_SetStopDec(short nAxisNum,double decSmoothStop,double decAbruptStop);
int GA_GetStopDec(short nAxisNum,double *pDecSmoothStop,double *pDecAbruptStop);
int GA_CtrlMode(short nAxisNum,short mode);
int GA_GetCtrlMode(short nAxisNum,short *pMode);
int GA_SetStopIo(short nAxisNum,short stopType,short inputType,short inputIndex);
int GA_SetAdcFilter(short nAdcNum,short nFilterTime);
int GA_GetAdcFilter(short nAdcNum,unsigned short* pFilterTime);
int GA_SetAdcBias(short nAdcNum,short nBias);
int GA_GetAdcBias(short nAdcNum,short *pBias);
int GA_SetArriveIo(short nAxisNum,short nCardIndex,short nIONum);
int GA_SetSmoothTime(short nAxisNum,short nSmoothTime);
int GA_SetIndexFliter(short nAxisNum,short nFilter);
int GA_AxisSMoveEnable(short nAxisNum,double dJ);
int GA_AxisSMoveDisable(short nAxisNum);
int GA_CrdSMoveEnable(short nAxisNum,double dJ);
int GA_CrdSMoveDisable(short nCrdNum);
int GA_SetCricleMode(short nAxisNum,int lCircleRange);
//运动状态检测指令列表
int GA_GetSts(short nAxisNum,int *pSts,short nCount=1,unsigned int *pClock=NULL);
int GA_ClrSts(short nAxisNum,short nCount=1);
int GA_GetPrfMode(short profile,int *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetPrfPos(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetPrfVel(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetPrfAcc(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetAxisPrfPos(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetAxisPrfVel(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetAxisPrfAcc(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetAxisEncPos(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetAxisEncVel(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetAxisEncAcc(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetAxisError(short nAxisNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_Stop(int lMask,int lOption);
int GA_StopEx(int lCrdMask,int lCrdOpion,int lAxisMask0,int lAxisOption0);
int GA_AxisOn(short nAxisNum);
int GA_AxisOff(short nAxisNum);
int GA_GetAllSysStatus(TAllSysStatusData *pAllSysStatusData);
int GA_GetAllSysStatusSX(TAllSysStatusDataSX *pAllSysStatusData);

//点位运动指令列表（包括点位和速度模式）
int GA_PrfTrap(short nAxisNum);
int GA_SetTrapPrm(short nAxisNum,TTrapPrm *pPrm);
int GA_SetTrapPrmSingle(short nAxisNum,double dAcc,double dDec,double dVelStart,short  dSmoothTime);
int GA_GetTrapPrm(short nAxisNum,TTrapPrm *pPrm);
int GA_GetTrapPrmSingle(short nAxisNum,double* dAcc,double* dDec,double* dVelStart,short*  dSmoothTime);
int GA_SetTrapPosAndUpdate(short nAxisNum,long long llPos,double dVel,double dAcc,double dDec,double dVelStart,short nSmoothTime,short nBlock);
int GA_PrfJog(short nAxisNum);
int GA_SetJogPrm(short nAxisNum,TJogPrm *pPrm);
int GA_SetJogPrmSingle(short nAxisNum,double dAcc,double dDec,double dSmooth);
int GA_GetJogPrm(short nAxisNum,TJogPrm *pPrm);
int GA_GetJogPrmSingle(short nAxisNum,double* dAcc,double* dDec,double* dSmooth);
int GA_SetPos(short nAxisNum,int pos);
int GA_GetPos(short nAxisNum,int *pPos);
int GA_SetVel(short nAxisNum,double vel);
int GA_GetVel(short nAxisNum,double *pVel);
int GA_SetMultiVel(short nAxisNum,double *pVel,short nCount=1);
int GA_SetMultiPos(short nAxisNum,int *pPos,short nCount=1);
int GA_Update(int mask);

//电子齿轮模式指令列表
int GA_PrfGear(short nAxisNum,short dir=0);
int GA_SetGearMaster(short nAxisNum,short nMasterAxisNum,short masterType=GEAR_MASTER_PROFILE);
int GA_GetGearMaster(short nAxisNum,short *nMasterAxisNum,short *pMasterType=NULL);
int GA_SetGearRatio(short nAxisNum,int masterEven,int slaveEven,int masterSlope=0,int lStopSmoothTime = 200);
int GA_GetGearRatio(short nAxisNum,int *pMasterEven,int *pSlaveEven,int *pMasterSlope=NULL,int *pStopSmoothTime=NULL);
int GA_GearStart(int mask);
int GA_GearStop(int lAxisMask,int lEMGMask);
int GA_SetGearEvent(short nAxisNum,short nEvent,double startPara0,double startPara1);
int GA_GetGearEvent(short nAxisNum,short *pEvent,double *pStartPara0,double *pStartPara1);
int GA_SetGearIntervalTime(short nAxisNum,short nIntervalTime);
int GA_GetGearIntervalTime(short nAxisNum,short *nIntervalTime);
int GA_GearSetMaxVel(short nAxisNum,double dMaxVel);

//电子凸轮模式指令列表
int GA_PrfCam(short nAxisNum,short nTableNum);
int GA_SetCamMaster(short nAxisNum,short nMasterAxisNum,short nMasterType);
int GA_GetCamMaster(short nAxisNum,short *pnMasterAxisNum,short *pMasterType);
int GA_SetCamEvent(short nAxisNum,short nEvent,double startPara0,double startPara1);
int GA_GetCamEvent(short nAxisNum,short *pEvent,double *pStartPara0,double *pStartPara1);
int GA_SetCamIntervalTime(short nAxisNum,short nIntervalTime);
int GA_GetCamIntervalTime(short nAxisNum,short *nIntervalTime);
int GA_SetUpCamTable(short nCamTableNum,int lMasterValueMax, int *plSlaveCamData, int lCamTableLen);
int GA_SetUpCamTableByKeyPoint(short nCamTableNum,int *plMasterKeyPoint, int *plSlaveKeyPoint, int lKeyPointLen);
int GA_GetCamTable(short nCamTableNum,int* plMasterValueMax, int *plSlaveCamData, int* plCamTableLen);
int GA_DownCamTable(short nTableNum,int *pProgress);
int GA_CamStart(int lMask);
int GA_CamStop(int lAxisMask,int lEMGMask);

//PT模式指令列表
int GA_PrfPt(short nAxisNum,short mode=PT_MODE_STATIC);
int GA_PtSpace(short nAxisNum,int *pSpace,short nCount);
int GA_PtRemain(short nAxisNum,int *pRemainSpace,short nCount);
int GA_PtData(short nAxisNum,short* pData,int lLength,double dDataID);
int GA_PtClear(int lAxisMask);
int GA_PtStart(int lAxisMask);

//插补运动模式指令列表
int GA_SetCrdPrm(short nCrdNum,TCrdPrm *pCrdPrm);
int GA_SetCrdPrmEX(short nCrdNum,TCrdPrmEx *pCrdPrm);
int GA_SetCrdPrmSingle(short nCrdNum,short dimension,short *profile,double synVelMax,double synAccMax,short evenTime,short setOriginFlag,int *originPos);
int GA_SetCrdPrmSingleEX(short nCrdNum,short dimension,short profile0,short profile1,short profile2,short profile3,short profile4,short profile5,short profile6,short profile7,double synVelMax,double synAccMax,short evenTime,short setOriginFlag,int originPos0,int originPos1,int originPos2,int originPos3,int originPos4,int originPos5,int originPos6,int originPos7);
int GA_GetCrdPrm(short nCrdNum,TCrdPrm *pCrdPrm);
int GA_SetAddAxis(short nAxisNum,short nAddAxisNum);
int GA_SetCrdOffset(short nCrdNum,int lOffsetX,int lOffsetY,int lOffsetZ,int lOffsetA,int lOffsetB,double dOffsetAngle);
int GA_InitLookAhead(short nCrdNum,short FifoIndex,TLookAheadPrm* plookAheadPara);
int GA_InitLookAheadSingle(short nCrdNum,short FifoIndex,int lookAheadNum,double* dSpeedMax,double* dAccMax,double *dMaxStepSpeed,double *dScale);
int GA_InitLookAheadSingleEX(short nCrdNum,short FifoIndex,int lookAheadNum,double dSpeedMax0,double dSpeedMax1,double dSpeedMax2,double dSpeedMax3,double dSpeedMax4,double dAccMax0,double dAccMax1,double dAccMax2,double dAccMax3,double dAccMax4,double dMaxStepSpeed0,double dMaxStepSpeed1,double dMaxStepSpeed2,double dMaxStepSpeed3,double dMaxStepSpeed4,double dScale0,double dScale1,double dScale2,double dScale3,double dScale4);
int GA_CrdClear(short nCrdNum,short FifoIndex);
int GA_LnX(short nCrdNum,int x,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum = 0);
int GA_LnXY(short nCrdNum,int x,int y,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum = 0);
int GA_LnXYZ(short nCrdNum,int x,int y,int z,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum = 0);
int GA_LnXYZA(short nCrdNum,int x,int y,int z,int a,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_LnXYZAB(short nCrdNum,int x,int y,int z,int a,int b,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_LnXYZABC(short nCrdNum,int x,int y,int z,int a,int b,int c,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_LnAll(short nCrdNum,int* pPos,short nDim, double synVel,double synAcc,double velEnd,short FifoIndex=0,int segNum = 0);
int GA_LnXYG0(short nCrdNum,int x,int y,double synVel,double synAcc,short FifoIndex=0,int segNum=-1);
int GA_LnXYZG0(short nCrdNum,int x,int y,int z,double synVel,double synAcc,short FifoIndex=0,int segNum = 0);
int GA_LnXYZAG0(short nCrdNum,int x,int y,int z,int a,double synVel,double synAcc,short FifoIndex=0,int segNum=-1);
int GA_LnXYZABG0(short nCrdNum,int x,int y,int z,int a,int b,double synVel,double synAcc,short FifoIndex=0,int segNum=-1);
int GA_LnXYZABCG0(short nCrdNum,int x,int y,int z,int a,int b,int c,double synVel,double synAcc,short FifoIndex=0,int segNum=-1);
int GA_LnXYZABMaskG0(short nCrdNum,int x,int y,int z,int a,int b,int lEnableMask,double synVel,double synAcc,short FifoIndex=0,int segNum=-1);
int GA_LnAllG0(short nCrdNum,int *pPos,short nDim,double synVel,double synAcc,short FifoIndex=0,int segNum=-1);
int GA_LnXYCmpPluse(short nCrdNum,int x,int y,double synVel,double synAcc,double velEnd,short nChannelMask,short nPluseType, short nTime,short nTimerFlag,short FifoIndex=0,int segNum=-1);
int GA_ArcXYC(short nCrdNum,int x,int y,double xCenter,double yCenter,short circleDir,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum = 0);
int GA_ArcXZC(short nCrdNum,int x,int z,double xCenter,double zCenter,short circleDir,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum = 0);
int GA_ArcYZC(short nCrdNum,int y,int z,double yCenter,double zCenter,short circleDir,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum = 0);
int GA_HelixXYCZ(short nCrdNum,int x,int y,int z,double xCenter,double yCenter,float k, short circleDir,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_HelixXZCY(short nCrdNum,int x,int y,int z,double xCenter,double zCenter,float k, short circleDir,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_HelixYZCX(short nCrdNum,int x,int y,int z,double yCenter,double zCenter,float k, short circleDir,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_HelixXYCCount(short nCrdNum,double xCenter,double yCenter,float k,float CirlceCount, short circleDir,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_HelixXZCCount(short nCrdNum,double xCenter,double zCenter,float k,float CirlceCount, short circleDir,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_HelixYZCCount(short nCrdNum,double yCenter,double zCenter,float k,float CirlceCount, short circleDir,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_EllipticXYC(short nCrdNum,int x,int y,double xCenter,double yCenter,short circleDir,int R1,int R2,double synVel,double synAcc,double velEnd=0,short FifoIndex=0,int segNum=-1);
int GA_BufIO(short nCrdNum,unsigned short nDoType,unsigned short nCardIndex,unsigned short doMask,unsigned short doValue,short FifoIndex=0,int segNum = 0);
int GA_BufIOReverse(short nCrdNum,unsigned short nDoType,unsigned short nCardIndex,unsigned short doMask,unsigned short doValue,unsigned short nReverseTime,short FifoIndex=0,int segNum = 0);
int GA_BufWaitIO(short nCrdNum,unsigned short nCardIndex,unsigned short nIOPortIndex,unsigned short nLevel,unsigned int lWaitTimeMS,unsigned short nFilterTime,short FifoIndex=0,int segNum=-1);
int GA_BufDelay(short nCrdNum,unsigned int ulDelayTime,short FifoIndex=0,int segNum = 0);
int GA_BufSetM(short nCrdNum,unsigned short nMAddr,unsigned short nMValue,short FifoIndex=0,int segNum=-1);
int GA_BufWaitM(short nCrdNum,unsigned short nMAddr,unsigned short nMValue,short FifoIndex=0,int segNum=-1);
int GA_BufCmpData(short nCrdNum,short nCmpEncodeNum,short nPluseType, short nStartLevel, short nTime,int *pBuf, short nBufLen,short nAbsPosFlag,short nTimerFlag,short nFifoIndex,int lSegNum);
int GA_BufCmpPluse(short nCrdNum,short nChannel,short nPluseType,short nTime,short nTimerFlag,short nFifoIndex,int lSegNum);
int GA_BufCmpRpt(short nCrdNum,short nCmpNum, unsigned int lIntervalTime, short nTime,short nTimeFlag,unsigned int ulRptTime,short FifoIndex=0,int segNum=-1);
int GA_BufMoveVel(short nCrdNum,short nAxisMask,float* pVel,short nFifoIndex=0,int lSegNum=-1);
int GA_BufMoveVelEX(short nCrdNum,short nAxisMask,float* pVel,short nFifoIndex=0,int lSegNum=-1);
int GA_BufMoveAcc(short nCrdNum,short nAxisMask,float* pAcc,short nFifoIndex=0,int lSegNum=-1);
int GA_BufMoveAccEX(short nCrdNum,short nAxisMask,float* pAcc,short nFifoIndex=0,int lSegNum=-1);
int GA_BufMoveDec(short nCrdNum,short nAxisMask,float* pDec,short nFifoIndex=0,int lSegNum=-1);
int GA_BufMoveDecEX(short nCrdNum,short nAxisMask,float* pDec,short nFifoIndex=0,int lSegNum=-1);
int GA_BufMove(short nCrdNum,short nAxisMask,int* pPos,short nModalMask,short nFifoIndex=0,int lSegNum=-1);
int GA_BufMoveEX(short nCrdNum,short nAxisMask,int* pPos,short nModalMask,short nFifoIndex=0,int lSegNum=-1);
int GA_BufGear(short nCrdNum,short nAxisMask,int* pPos,short nFifoIndex=0,int lSegNum=-1);
int GA_BufJog(short nCrdNum,short nAxisNum,double dAccDec,double dVel,short nBlock,short nFifoIndex,int lUserSegNum);
int GA_BufZeroPos(short nCrdNum,short nAxisNum,short nFifoIndex,int lUserSegNum=-1);
int GA_GetCrdErrStep(short nCrdNum,unsigned int *pulErrStep);
//GA_BufPWM为激光卡专用函数
int GA_BufPWM(short nCrdNum,short nPwmNum ,double dFreq,double dDuty,short nFifoIndex,int lUserSegNum=-1);
//GA_BufDA为激光卡专用函数
int GA_BufDA(short nCrdNum,short nDacNum,short nValue,short nFifoIndex,int lUserSegNum=-1);
int GA_BufABSMode(short nCrdNum,short nFifoIndex,int lUserSegNum=-1);
int GA_BufRELMode(short nCrdNum,short nFifoIndex,int lUserSegNum=-1);

int GA_CrdDataMaxLine(unsigned int ulCrdDataMaxLine);
int GA_CrdMaxLinePerFrame(unsigned int ulCrdMaxLinePerFrame);
int GA_CrdData(short nCrdNum,void *pCrdData,short FifoIndex=0);
int GA_CrdStart(short mask,short option);
int GA_SetOverride(short nCrdNum,double synVelRatio);
int GA_G00SetOverride(short nCrdNum,double synVelRatio);
int GA_SetCrdStopDec(short nCrdNum,double decSmoothStop,double decAbruptStop);
int GA_GetCrdPos(short nCrdNum,double *pPos);
int GA_GetCrdVel(short nCrdNum,double *pSynVel);
int GA_CrdSpace(short nCrdNum,int *pSpace,short FifoIndex=0);
int GA_CrdStatus(short nCrdNum,short *pCrdSts,int *pSegment,short FifoIndex=0);
int GA_SetUserSegNum(short nCrdNum,int segNum,short FifoIndex=0);
int GA_GetUserSegNum(short nCrdNum,int *pSegment,short FifoIndex=0);
int GA_GetRemainderSegNum(short nCrdNum,int *pSegment,short FifoIndex=0);
int GA_GetLookAheadSpace(short nCrdNum,int *pSpace,short nFifoIndex=0);
int GA_GetLookAheadSegCount(short nCrdNum,int *pSegCount,short FifoIndex=0);
int GA_StartDebugLog(short nDir=0);

//访问硬件资源指令列表
int GA_GetDi(short nDiType,int *pValue);
int GA_GetDiRaw(short nDiType,int *pValue);
int GA_GetDiReverseCount(short nDiType,short diIndex,unsigned int *pReverseCount,short nCount=1);
int GA_SetDiReverseCount(short nDiType,short diIndex,unsigned int ReverseCount,short nCount=1);
int GA_SetDo(short nDoType,int value);
int GA_SetDoBit(short nDoType,short nDoNum,short value);
int GA_SetDoBitReverse(short nDoType,short nDoNum,short nValue,short nReverseTime);
int GA_SetDoBitReverseEx(unsigned short nCardIndex,short nDoType,short nDoNum,short nValue,short nReverseTime);
int GA_GetDo(short nDoType,int *pValue);
int GA_GetEncPos(short nEncodeNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetEncVel(short nEncodeNum,double *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_SetEncPos(short nEncodeNum,int encPos);
int GA_GetMPGVel(double *pValue,unsigned int *pClock=NULL);
int GA_SetMPGPos(int lMPGPos);
int GA_SetDac(short nDacNum,short* pValue,short nCount=1);
int GA_GetDac(short nDacNum,short *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetAdc(short nADCNum,short *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_SetPwm(short nPwmNum ,double dFreq,double dDuty);
int GA_GetPwm(short nPwmNum ,double *pFreq,double *pDuty);
int GA_SetExtDoValue(short nCardIndex,unsigned int *value,short nCount=1);
int GA_GetExtDiValue(short nCardIndex,unsigned int *pValue,short nCount=1);
int GA_GetExtDoValue(short nCardIndex,unsigned int *pValue,short nCount=1);
int GA_SetExtDiFilter(short nCardIndex,short FilterTime,int ulIOMask);
int GA_SetExtDoBit(short nCardIndex,short nBitIndex,unsigned short nValue);
int GA_GetExtDiBit(short nCardIndex,short nBitIndex,unsigned short *pValue);
int GA_GetExtDoBit(short nCardIndex,short nBitIndex,unsigned short *pValue);
int GA_SendEthToUartString(short nUartNum,unsigned char*pSendBuf, short nLength);
int GA_ReadUartToEthString(short nUartNum,unsigned char* pRecvBuf, short* pLength);
int GA_ReadUartToEthSingle(short nUartNum,unsigned char* pRecvByte1, unsigned char* pRecvByte2,unsigned char* pRecvByte3,unsigned char* pRecvByte4,unsigned char* pRecvByte5,unsigned char* pRecvByte6,unsigned char* pRecvByte7,unsigned char* pRecvByte8,unsigned char* pRecvByte9,unsigned char* pRecvByte10,unsigned char* pRecvByte11,unsigned char* pRecvByte12,unsigned char* pRecvByte13,unsigned char* pRecvByte14,unsigned char* pRecvByte15,unsigned char* pRecvByte16,short* pLength);
int GA_SetExDac(short nCardIndex,short nDacNum,short* pValue,short nCount=1);
int GA_GetExDac(short nCardIndex,short nDacNum,short *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_GetExAdc(short nCardIndex,short nADCNum,short *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_UartConfig(unsigned short nUartNum,	unsigned int uLBaudRate,unsigned short nDataLength,unsigned short nVerifyType,unsigned short nStopBitLen);
int GA_SetIOTrans(short nEnable,short nCardIndex,short nInPortIndex,short nEdge,short nOutPortIndex,short nLevel,unsigned int ulTime,unsigned short nSurvivalTime);
int GA_SetIOTrigPluse(short nEnable,short nCardIndex,short nIOPortIndex,short nEdge,short nAxisNum,double dAcc,double dVel,unsigned int ulPluse,unsigned short nSurvivalTime);
int GA_GpioCmpBufData(short nCmpGpoIndex,short nCmpSource, short nPluseType, short nStartLevel, short nTime, short nTimerFlag,short nAbsPosFlag,short nBufLen,int *pBuf);
int GA_GpioCmpBufStop(int *pGpioMask,short nCount);
//比较输出指令
int GA_CmpPluse(short nChannelMask, short nPluseType1, short nPluseType2, short nTime1,short nTime2, short nTimeFlag1, short nTimeFlag2);
int GA_CmpBufSetChannel(short nBuf1ChannelNum,short nBuf2ChannelNum);
int GA_CmpBufData(short nCmpEncodeNum, short nPluseType, short nStartLevel, short nTime, int *pBuf1, short nBufLen1, int *pBuf2, short nBufLen2,short nAbsPosFlag=0,short nTimerFlag=0);
int GA_CmpBufSts(short *pStatus,unsigned short *pCount1,unsigned short *pCount2);
int GA_CmpBufStop(short nChannel);
int GA_CmpRpt(short nCmpNum, unsigned int lIntervalTime, short nTime,short nTimeFlag,unsigned int ulRptTime);
int GA_CmpBufRpt(short nEncNum,short nDir,short nEncFlag,int lTrigValue,short nCmpNum, unsigned int lIntervalTime, short nTime,short nTimeFlag,unsigned int ulRptTime);
int GA_CmpSetDynamic(short nCrdNum, double dMinSpeed,double dMaxSpeed,unsigned short nMinX,unsigned short nMaxX,unsigned short nMinY,unsigned short nMaxY,unsigned short nEnable);
int GA_CmpGetDynamic(short nCrdNum, double* dMinSpeed,double* dMaxSpeed,unsigned short* nMinX,unsigned short* nMaxX,unsigned short* nMinY,unsigned short* nMaxY,unsigned short* nEnable);
int GA_CmpRstFpgaCount(unsigned int ulMask);
int GA_CmpGetFpgaCount(unsigned short *pFPGACount,unsigned short nCount);

//高速硬件捕获指令列表
int GA_SetCaptureMode(short nEncodeNum,short mode);
int GA_GetCaptureMode(short nEncodeNum,short *pMode,short nCount=1);
int GA_GetCaptureStatus(short nEncodeNum,short *pStatus,int *pValue,short nCount=1,unsigned int *pClock=NULL);
int GA_SetCaptureSense(short nEncodeNum,short mode,short sense);
int GA_GetCaptureSense(short nEncodeNum,short mode,short *sense);
int GA_ClearCaptureStatus(short nEncodeNum);
int GA_SetContinueCaptureMode(short nEncodeNum,short nMode,short nContinueMode,short nFilterTime);
int GA_GetContinueCaptureData(short nEncodeNum,int *pCapturePos,short* pCaptureCount);
int GA_SetCaptureSource(short nAxisNum,short nSource);

//安全机制指令列表
int GA_SetSoftLimit(short nAxisNum,int lPositive,int lNegative);
int GA_GetSoftLimit(short nAxisNum,int *pPositive,int *pNegative);
int GA_SetHardLimP(short nAxisNum,short nType ,short nCardIndex,short nIOIndex);
int GA_SetHardLimN(short nAxisNum,short nType ,short nCardIndex,short nIOIndex);
int GA_EStopSetIO(short nCardIndex,short nIOIndex,short nEStopSns,unsigned int lFilterTime);
int GA_EStopConfig(unsigned int ulEnableMask,unsigned int ulEnableValue,short nAdcMask,short nAdcValue,unsigned int ulIOMask,unsigned int ulIOValue);
int GA_EStopOnOff(short nEStopOnOff);
int GA_EStopGetSts(short *nEStopSts);
int GA_EStopClrSts();
int GA_CrdHlimEnable(short nCrdNum ,short nEnableFlag);
int GA_SoftLimEnc(short nAxisNum,short nEnableFlag);

//自动回零相关API
int GA_HomeStart(int iAxisNum);
int GA_HomeStop(int iAxisNum);
int GA_HomeSetPrm(int iAxisNum,TAxisHomePrm *pAxisHomePrm);
int GA_HomeSetPrmSingle(short iAxisNum,short nHomeMode,short nHomeDir,int lOffset,double dHomeRapidVel,double dHomeLocatVel,double dHomeIndexVel,double dHomeAcc,int lHomeIndexDis,int lHomeBackDis,short nDelayTimeBeforeZero);
int GA_HomeGetPrm(int iAxisNum,TAxisHomePrm *pAxisHomePrm);
int GA_HomeGetPrmSingle(short iAxisNum,short *nHomeMode,short *nHomeDir,int *lOffset,double* dHomeRapidVel,double* dHomeLocatVel,double* dHomeIndexVel,double* dHomeAcc);
int GA_HomeGetSts(short nAxisNum,unsigned short* pStatus,int* pHomeLocateAbsPos,int* pZCaptureAbsPos,int* pZCaptureDisToSensor);
int GA_HomeGetFailReason(short nAxisNum,unsigned short* pFailReason);
int GA_HomeSetFinishFlag(short nAxisNum,unsigned short nValue);

//EtherCAT相关API
int GA_ECatInit();
int GA_ECatGetInitStep(short* pCutInitSlaveNum,short* pMode,short* pModeStep,short* pPauseStatus);
int GA_ECatGetSlaveCount(short* pCount);
int GA_ECatSetPluseAxisNum(short* pAxisNum,short nCount);
int GA_ECatSetAdoValue(short nStationNum,short nAdoAddr,short nAdoValue);
int GA_ECatGetAdoValue(short nAxisNum,short nAdoAddr,short *pAdoValue);
int GA_ECatSetSdoValue(short nStationNum,short nSdoIndex,short nSdoSubIndex,int lSdoValue,short nLen);
int GA_ECatGetSdoValue(short nStationNum,short nSdoIndex,short nSdoSubIndex,int *pSdoValue,short *pPdoFlag,short nLen,short nSignFlag);
int GA_ECatSetProbeCaptureStart(short nStationNum,short nProbeNum,short nProbeSource,short nProbeSense,short nContinueFlag,short nAutoStopFlag);
int GA_ECatGetProbeCaptureStatus(short nStationNum,short nProbeNum,short* nSatus,int *pValueP,int *pValueF);
int GA_ECatSetPDOConfig(short nStationNum,short nGroupNum,TECatPDOParm* pEcatPrm);
int GA_ECatGetPDOConfig(short nStationNum,short nGroupNum,TECatPDOParm* pEcatPrm);
int GA_ECatLoadPDOConfig(short nStationNum);
int GA_ECatResetPDOConfig(short nStationNum);
int GA_ECatHomeStart(short nStationNum,short nHomeMode,double dHomeRapidVel,double dHomeLocatVel,double dHomeAcc,int lOffset,unsigned short nDelayTime);
int GA_ECatHomeStop(short nStationNum);
int GA_ECatSetCtrlMode(short nStationNum,unsigned short nCtrlMode);
int GA_ECatGetStatusWord(short nStationNum,short *pEcatStatusValue);
int GA_ECatSetAllPDOData(short nStationNum,unsigned char* pData,unsigned short nLen);
int GA_ECatGetAllPDOData(short nStationNum,unsigned char* pData,unsigned short* pLen);
int GA_ECatSetPlusePerCircle(short nAxisNum,long long lPlusePerCircleOrg,long long lPlusePerCircle);
int GA_ECatSetCtrlBit(short nStationNum,unsigned int ulMask,unsigned int ulValue);

int GA_ECatSetOrgPosAbs(short nStationNum,int lOrgPosAbs);
int GA_ECatSetOrgPosCur(short nStationNum);
int GA_ECatGetOrgPosAbs(short nStationNum,int* plOrgPosAbs);
int GA_ECatLoadOrgPosAbs(short nStationNum);
int GA_ECatGetDCOffset(int *lDCOffset);
int GA_ECatSetDCOffset(short nStationNum,int lDCOffset);
int GA_ECatSetPauseWhenInit(short nPauseStep);

//手轮相关
int GA_StartHandwheel(short nAxisNum,short nMasterAxisNum = 9,int lMasterEven = 1,int lSlaveEven = 1,short nIntervalTime = 0,double dAcc = 0.1,double dDec = 0.1,double dVel = 50,short nStopWaitTime = 0);
int GA_EndHandwheel(short nAxisNum);
int GA_EndHandwheelSmooth(short nAxisNum);

//激光相关
int GA_LaserPowerMode(short nChannelIndex,short nPowerMode,double dMaxValue,double dMinValue,short nDelayMode);
int GA_LaserSetPower(short nChannelIndex,double dPower);
int GA_LaserOn(short nChannelIndex);
int GA_LaserOff(short nChannelIndex);
int GA_LaserGetPowerAndOnOff(short nChannelIndex,double* dPower,short* pOnOff);
int GA_LaserFollowRatio(short nChannelIndex,double dMinSpeed,double dMaxSpeed,double dMinPower,double dMaxPower,short nFifoIndex);

//Robot相关API
int GA_RobotSetPrm(unsigned short RobotID,unsigned int ulRobotType,short nJogAxisCount,short *pJogAxisList,short nVirAxisCount,short* pVirAxisList,void *RobotParm);
int GA_RobotSetPrmDelta40001(unsigned short RobotID,short nJogAxisCount,short JogAxis1,short JogAxis2,short JogAxis3,short nVirAxisCount,short VirAxisX,short VirAxisY,short VirAxisZ,int lPlusePerCircle,double dRotateAngle1,double dRotateAngle2,double dRotateAngle3,double dDisFixPlatform,double dLengthArm1,double dLengthArm2,double dDisMovPlatform);
int GA_RobotSetForward(unsigned short nRobotID);
int GA_RobotSetInverse(unsigned short nRobotID);
int GA_RobotSixArmCaculate(unsigned short nRobotID,short nPointCount,int* lAxisPos1,int* lAxisPos2,int* lAxisPos3,int* lAxisPos4,int* lAxisPos5,int* lAxisPos6,double* dX,double* dY,double* dZ);
//其他
int GA_StartLog();
int GA_GetIP(unsigned int* pIP);
int GA_SetIP(unsigned int ulIP);
int GA_GetID(unsigned int* pID);
int GA_SetPLCShortD(int lAdd,short *pData,short nCount);
int GA_GetPLCShortD(int lAdd,short *pData,short nCount);
int GA_SetPLCintD(int lAdd,int *pData,short nCount);
int GA_GetPLCintD(int lAdd,int *pData,short nCount);
int GA_SetPLCFloatD(int lAdd,float *pData,short nCount);
int GA_GetPLCFloatD(int lAdd,float *pData,short nCount);
int GA_SetPLCM(int lAdd,char *pData,short nCount);
int GA_GetPLCM(int lAdd,char *pData,short nCount);
int GA_WriteInterFlash(unsigned char* pData,short nLength);
int GA_ReadInterFlash(unsigned char*pData,short nLength);
int GA_DownPitchErrorTable(short nTableNum,short nPointNum,int lStartPos,int lEndPos,short *pErrValue1,short *pErrValue2);
int GA_ReadPitchErrorTable(short nTableNum,short* pPointNum,int* pStartPos,int* pEndPos,short *pErrValue1,short *pErrValue2);
int GA_AxisErrPitchOn(short nAxisNum);
int GA_AxisErrPitchOff(short nAxisNum);
int GA_SetKeepAlive(int AliveTime);
int GA_StartWatch(int lAxisMask,int lPackageCountFlag,int lUserSegNumFlag,int lReserve,char *FilePath);
int GA_StopWatch();
int GA_StartHighSpeedCommu(short nLevel);
int GA_GetDllVersion();
int GA_GetCardMessage(char *cMessage);
int GA_ClrCardMessage();

#endif // GAS_LINUX_N_H
