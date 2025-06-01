import asyncio
import logging
from system.system_manager import SystemManager

async def main():
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 시스템 매니저 초기화
        system_manager = SystemManager()
        
        # 시스템 시작 (예: 삼성전자, 1시간봉)
        await system_manager.start_system(symbol="005930", timeframe="1h")
        
        # 시스템이 계속 실행되도록 유지
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("사용자에 의해 시스템이 중지되었습니다.")
        await system_manager.stop_system()
        logger.info("시스템 종료됨")
        exit(0)
    except Exception as e:
        logger.error(f"시스템 실행 중 오류 발생: {str(e)}")
        await system_manager.stop_system()
        logger.info("시스템 종료됨")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 