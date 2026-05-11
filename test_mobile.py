"""
移动端功能测试
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import WorkMode, RiskLevel, Direction


class TestConfig(unittest.TestCase):
    """测试配置模块"""

    def test_work_modes(self):
        self.assertEqual(WorkMode.FIND, "find")
        self.assertEqual(WorkMode.GRAB, "grab")
        self.assertEqual(WorkMode.WALK, "walk")
        self.assertEqual(WorkMode.READ, "read")
        self.assertEqual(WorkMode.ASK, "ask")

    def test_risk_levels(self):
        self.assertEqual(RiskLevel.P0, "P0")
        self.assertEqual(RiskLevel.P1, "P1")
        self.assertEqual(RiskLevel.P2, "P2")
        self.assertEqual(RiskLevel.P3, "P3")

    def test_directions(self):
        self.assertEqual(Direction.LEFT, "left")
        self.assertEqual(Direction.CENTER, "center")
        self.assertEqual(Direction.RIGHT, "right")


class TestPlatformAdapter(unittest.TestCase):
    """测试平台适配器"""

    def test_import(self):
        from platform_adapter import CameraAdapter, VoiceAdapter, VibrationAdapter
        self.assertTrue(callable(CameraAdapter))
        self.assertTrue(callable(VoiceAdapter))
        self.assertTrue(callable(VibrationAdapter))

    def test_camera_creation(self):
        from platform_adapter import CameraAdapter
        camera = CameraAdapter(camera_id=0, width=640, height=480)
        self.assertIsNotNone(camera)
        self.assertFalse(camera.is_opened)

    def test_voice_creation(self):
        from platform_adapter import VoiceAdapter
        voice = VoiceAdapter()
        self.assertIsNotNone(voice)

    def test_vibration_creation(self):
        from platform_adapter import VibrationAdapter
        vibration = VibrationAdapter()
        self.assertIsNotNone(vibration)

    def test_speech_recognizer_creation(self):
        from platform_adapter import SpeechRecognizerAdapter
        recognizer = SpeechRecognizerAdapter()
        self.assertIsNotNone(recognizer)


class TestAgentStateMachine(unittest.TestCase):
    """测试状态机"""

    def test_import(self):
        from agent.state_machine import AgentStateMachine
        self.assertTrue(callable(AgentStateMachine))

    def test_creation(self):
        from agent.state_machine import AgentStateMachine
        agent = AgentStateMachine()
        self.assertIsNotNone(agent)
        self.assertEqual(agent.state.current_mode, WorkMode.WALK)

    def test_set_mode(self):
        from agent.state_machine import AgentStateMachine
        agent = AgentStateMachine()
        agent.set_mode(WorkMode.FIND, "杯子")
        self.assertEqual(agent.state.current_mode, WorkMode.FIND)
        self.assertEqual(agent.state.target_object, "杯子")

    def test_handle_voice_command(self):
        from agent.state_machine import AgentStateMachine
        agent = AgentStateMachine()
        response = agent.handle_voice_command("帮我找杯子")
        self.assertIsNotNone(response)
        self.assertIn("杯子", response)


class TestPerceptionModules(unittest.TestCase):
    """测试感知模块"""

    def test_ocr_import(self):
        from perception.ocr import OCRModule
        self.assertTrue(callable(OCRModule))

    def test_object_detection_import(self):
        from perception.object_detection import ObjectDetectionModule
        self.assertTrue(callable(ObjectDetectionModule))

    def test_hand_tracking_import(self):
        from perception.hand_tracking import HandTrackingModule
        self.assertTrue(callable(HandTrackingModule))

    def test_depth_estimation_import(self):
        from perception.depth_estimation import DepthEstimationModule
        self.assertTrue(callable(DepthEstimationModule))


class TestSceneModules(unittest.TestCase):
    """测试场景模块"""

    def test_spatial_analyzer_import(self):
        from scene.spatial_analyzer import SpatialAnalyzer
        self.assertTrue(callable(SpatialAnalyzer))

    def test_risk_assessor_import(self):
        from scene.risk_assessor import RiskAssessor
        self.assertTrue(callable(RiskAssessor))

    def test_spatial_analyzer_creation(self):
        from scene.spatial_analyzer import SpatialAnalyzer
        analyzer = SpatialAnalyzer()
        self.assertIsNotNone(analyzer)

    def test_risk_assessor_creation(self):
        from scene.risk_assessor import RiskAssessor
        assessor = RiskAssessor()
        self.assertIsNotNone(assessor)


class TestAgentModes(unittest.TestCase):
    """测试 Agent 模式"""

    def test_find_mode(self):
        from agent.modes import FindMode
        mode = FindMode()
        self.assertIsNotNone(mode)

    def test_grab_mode(self):
        from agent.modes import GrabMode
        mode = GrabMode()
        self.assertIsNotNone(mode)
        self.assertEqual(mode.state, "searching")

    def test_walk_mode(self):
        from agent.modes import WalkMode
        mode = WalkMode()
        self.assertIsNotNone(mode)

    def test_read_mode(self):
        from agent.modes import ReadMode
        mode = ReadMode()
        self.assertIsNotNone(mode)

    def test_ask_mode(self):
        from agent.modes import AskMode
        mode = AskMode()
        self.assertIsNotNone(mode)


class TestMainApp(unittest.TestCase):
    """测试主应用"""

    def test_import(self):
        from main import GuideVisionApp
        self.assertTrue(callable(GuideVisionApp))


class TestSpatialAnalyzer(unittest.TestCase):
    """测试空间分析器"""

    def test_analyze_empty(self):
        from scene.spatial_analyzer import SpatialAnalyzer
        analyzer = SpatialAnalyzer()
        scene = analyzer.analyze([])
        self.assertIsNotNone(scene)
        self.assertEqual(len(scene.objects), 0)
        self.assertEqual(scene.risk_level, "safe")

    def test_analyze_with_objects(self):
        from scene.spatial_analyzer import SpatialAnalyzer
        from perception.object_detection import DetectedObject
        analyzer = SpatialAnalyzer()

        obj = DetectedObject(
            class_name="cup",
            class_name_cn="杯子",
            confidence=0.9,
            x1=100, y1=100, x2=200, y2=200,
        )
        scene = analyzer.analyze([obj])
        self.assertEqual(len(scene.objects), 1)
        self.assertEqual(scene.objects[0].name_cn, "杯子")


class TestRiskAssessor(unittest.TestCase):
    """测试风险评估器"""

    def test_assess_safe(self):
        from scene.risk_assessor import RiskAssessor
        from scene.spatial_analyzer import SceneGraph
        assessor = RiskAssessor()
        scene = SceneGraph()
        risk = assessor.assess(scene)
        self.assertIsNotNone(risk)
        self.assertEqual(risk.level, "P3")


if __name__ == "__main__":
    unittest.main()
