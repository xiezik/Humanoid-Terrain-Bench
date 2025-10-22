#!/usr/bin/env python3
"""
BEAMDOJO统一验证执行器
运行所有验证测试并生成综合报告
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class BeamDojoTestRunner:
    """BEAMDOJO测试运行器"""
    
    def __init__(self):
        self.project_root = "/home/cft/zikang/Humanoid-Terrain-Bench"
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PENDING',
            'test_suites': {},
            'summary': {}
        }
        
        print("BEAMDOJO 综合验证测试套件")
        print("="*50)
    
    def run_all_tests(self) -> bool:
        """运行所有测试套件"""
        success = True
        
        # 测试套件定义
        test_suites = [
            {
                'name': 'validation',
                'description': '功能验证测试',
                'script': 'scripts/validate_beamdojo.py',
                'args': ['--test-all'],
                'critical': True
            },
            {
                'name': 'unit_tests',
                'description': '单元测试',
                'script': 'tests/test_beamdojo_units.py',
                'args': [],
                'critical': True
            },
            {
                'name': 'integration_tests',
                'description': '集成测试',
                'script': 'tests/test_beamdojo_integration.py',
                'args': [],
                'critical': True
            },
            {
                'name': 'benchmark',
                'description': '性能基准测试',
                'script': 'scripts/benchmark_beamdojo.py',
                'args': ['--quick'],
                'critical': False
            }
        ]
        
        # 运行每个测试套件
        for suite in test_suites:
            print(f"\n{'='*20} {suite['description']} {'='*20}")
            
            suite_success = self._run_test_suite(suite)
            
            if not suite_success:
                if suite['critical']:
                    success = False
                    print(f"❌ 关键测试套件 '{suite['name']}' 失败")
                else:
                    print(f"⚠️  非关键测试套件 '{suite['name']}' 失败")
            else:
                print(f"✅ 测试套件 '{suite['name']}' 通过")
        
        # 生成综合报告
        self._generate_comprehensive_report(success)
        
        return success
    
    def _run_test_suite(self, suite: Dict[str, Any]) -> bool:
        """运行单个测试套件"""
        script_path = os.path.join(self.project_root, suite['script'])
        
        if not os.path.exists(script_path):
            print(f"  ❌ 测试脚本不存在: {script_path}")
            self.test_results['test_suites'][suite['name']] = {
                'status': 'SCRIPT_NOT_FOUND',
                'error': f"Script not found: {script_path}"
            }
            return False
        
        try:
            # 构建命令
            cmd = [sys.executable, script_path] + suite['args']
            
            print(f"  🚀 运行命令: {' '.join(cmd)}")
            
            # 执行测试
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            end_time = time.time()
            
            # 记录结果
            suite_result = {
                'status': 'PASSED' if result.returncode == 0 else 'FAILED',
                'return_code': result.returncode,
                'duration': end_time - start_time,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            self.test_results['test_suites'][suite['name']] = suite_result
            
            # 打印输出摘要
            if result.returncode == 0:
                print(f"  ✅ 测试通过 (耗时: {suite_result['duration']:.1f}秒)")
                # 提取关键信息
                self._extract_test_summary(suite['name'], result.stdout)
            else:
                print(f"  ❌ 测试失败 (返回码: {result.returncode})")
                print(f"  错误输出: {result.stderr[:200]}...")
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ 测试超时 (超过300秒)")
            self.test_results['test_suites'][suite['name']] = {
                'status': 'TIMEOUT',
                'error': 'Test execution timeout'
            }
            return False
            
        except Exception as e:
            print(f"  💥 测试执行异常: {e}")
            self.test_results['test_suites'][suite['name']] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def _extract_test_summary(self, suite_name: str, stdout: str):
        """从输出中提取测试摘要"""
        try:
            lines = stdout.split('\n')
            
            if suite_name == 'validation':
                # 提取验证测试摘要
                for line in lines:
                    if 'Tests Passed:' in line:
                        passed = int(line.split(':')[1].strip())
                        self.test_results['test_suites'][suite_name]['tests_passed'] = passed
                    elif 'Tests Failed:' in line:
                        failed = int(line.split(':')[1].strip())
                        self.test_results['test_suites'][suite_name]['tests_failed'] = failed
                    elif 'Success Rate:' in line:
                        rate = float(line.split(':')[1].strip().replace('%', ''))
                        self.test_results['test_suites'][suite_name]['success_rate'] = rate
            
            elif suite_name == 'unit_tests':
                # 提取单元测试摘要
                for line in lines:
                    if 'Ran' in line and 'tests in' in line:
                        parts = line.split()
                        if len(parts) >= 2:
                            num_tests = int(parts[1])
                            self.test_results['test_suites'][suite_name]['total_tests'] = num_tests
                    elif line.strip() == 'OK':
                        self.test_results['test_suites'][suite_name]['all_passed'] = True
            
            elif suite_name == 'integration_tests':
                # 提取集成测试摘要
                for line in lines:
                    if 'Tests Passed:' in line:
                        passed = int(line.split(':')[1].strip())
                        self.test_results['test_suites'][suite_name]['tests_passed'] = passed
                    elif 'Tests Failed:' in line:
                        failed = int(line.split(':')[1].strip())
                        self.test_results['test_suites'][suite_name]['tests_failed'] = failed
            
            elif suite_name == 'benchmark':
                # 提取基准测试摘要
                for line in lines:
                    if 'Forward Pass Overhead:' in line:
                        overhead = float(line.split(':')[1].strip().replace('%', ''))
                        self.test_results['test_suites'][suite_name]['forward_overhead'] = overhead
                    elif 'Memory Overhead:' in line:
                        overhead = float(line.split(':')[1].strip().replace('%', ''))
                        self.test_results['test_suites'][suite_name]['memory_overhead'] = overhead
                    elif 'Performance Improvement:' in line:
                        improvement = float(line.split(':')[1].strip().replace('%', ''))
                        self.test_results['test_suites'][suite_name]['performance_improvement'] = improvement
        
        except Exception as e:
            print(f"  ⚠️  无法提取 {suite_name} 摘要: {e}")
    
    def _generate_comprehensive_report(self, overall_success: bool):
        """生成综合报告"""
        self.test_results['overall_status'] = 'PASSED' if overall_success else 'FAILED'
        
        # 计算汇总统计
        total_suites = len(self.test_results['test_suites'])
        passed_suites = sum(1 for suite in self.test_results['test_suites'].values() 
                           if suite.get('status') == 'PASSED')
        
        self.test_results['summary'] = {
            'total_suites': total_suites,
            'passed_suites': passed_suites,
            'failed_suites': total_suites - passed_suites,
            'success_rate': (passed_suites / total_suites * 100) if total_suites > 0 else 0
        }
        
        # 保存详细报告
        report_path = os.path.join(
            self.project_root, 
            f"comprehensive_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        # 打印综合摘要
        self._print_comprehensive_summary(report_path)
    
    def _print_comprehensive_summary(self, report_path: str):
        """打印综合摘要"""
        print(f"\n{'='*60}")
        print("BEAMDOJO 综合验证报告")
        print("="*60)
        
        summary = self.test_results['summary']
        print(f"整体状态: {'✅ 通过' if self.test_results['overall_status'] == 'PASSED' else '❌ 失败'}")
        print(f"测试套件: {summary['passed_suites']}/{summary['total_suites']} 通过")
        print(f"成功率: {summary['success_rate']:.1f}%")
        
        print(f"\n详细结果:")
        for suite_name, suite_result in self.test_results['test_suites'].items():
            status_icon = "✅" if suite_result.get('status') == 'PASSED' else "❌"
            duration = suite_result.get('duration', 0)
            print(f"  {status_icon} {suite_name}: {suite_result.get('status', 'UNKNOWN')} ({duration:.1f}s)")
            
            # 显示额外信息
            if 'tests_passed' in suite_result:
                total = suite_result.get('tests_passed', 0) + suite_result.get('tests_failed', 0)
                print(f"      └─ 通过: {suite_result.get('tests_passed', 0)}/{total}")
            
            if 'forward_overhead' in suite_result:
                print(f"      └─ 前向开销: {suite_result['forward_overhead']:.1f}%")
            
            if 'performance_improvement' in suite_result:
                print(f"      └─ 性能提升: {suite_result['performance_improvement']:.1f}%")
        
        print(f"\n关键指标:")
        
        # 功能正确性
        validation_suite = self.test_results['test_suites'].get('validation', {})
        if validation_suite.get('success_rate'):
            print(f"  功能验证成功率: {validation_suite['success_rate']:.1f}%")
        
        # 性能指标
        benchmark_suite = self.test_results['test_suites'].get('benchmark', {})
        if 'forward_overhead' in benchmark_suite:
            overhead = benchmark_suite['forward_overhead']
            status = "✅ 可接受" if overhead < 100 else "⚠️  需要优化"
            print(f"  双Critic前向开销: {overhead:.1f}% ({status})")
        
        if 'performance_improvement' in benchmark_suite:
            improvement = benchmark_suite['performance_improvement']
            status = "✅ 显著提升" if improvement > 5 else "⚠️  提升有限"
            print(f"  性能改进: {improvement:.1f}% ({status})")
        
        print(f"\n建议:")
        if self.test_results['overall_status'] == 'PASSED':
            print("  ✅ BEAMDOJO功能验证完全通过，可以投入使用")
        else:
            print("  ❌ 存在测试失败，请检查具体问题后重新测试")
        
        # 性能建议
        if 'forward_overhead' in benchmark_suite:
            overhead = benchmark_suite['forward_overhead']
            if overhead > 100:
                print("  ⚠️  双Critic开销较高，考虑在性能敏感场景中权衡使用")
            else:
                print("  ✅ 双Critic性能开销在可接受范围内")
        
        print(f"\n详细报告已保存至: {report_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BEAMDOJO综合验证测试')
    parser.add_argument('--quick', action='store_true', help='快速测试模式（跳过基准测试）')
    
    args = parser.parse_args()
    
    # 检查环境
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA设备: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch未安装，请先安装依赖")
        return False
    
    # 创建测试运行器
    runner = BeamDojoTestRunner()
    
    # 快速模式跳过基准测试
    if args.quick:
        print("🚀 快速模式：跳过基准测试")
        # 这里可以修改测试套件列表
    
    # 运行所有测试
    success = runner.run_all_tests()
    
    return success


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 测试执行过程中发生异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)