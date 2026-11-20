#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/go2/sport/sport_client.hpp>

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <networkInterface> [vx vy vyaw duration_sec]" << std::endl;
    return -1;
  }

  const std::string interface_name = argv[1];
  const float vx = (argc >= 3) ? std::stof(argv[2]) : 0.2F;
  const float vy = (argc >= 4) ? std::stof(argv[3]) : 0.0F;
  const float vyaw = (argc >= 5) ? std::stof(argv[4]) : 0.0F;
  const int duration_sec = (argc >= 6) ? std::stoi(argv[5]) : 3;

  unitree::robot::ChannelFactory::Instance()->Init(0, interface_name);

  unitree::robot::go2::SportClient sport_client;
  sport_client.SetTimeout(10.0F);
  sport_client.Init();
  sport_client.WaitLeaseApplied();

  // Single-entry flow: prepare standing state, then run motion command.
  std::cout << "[demo] prepare standing state..." << std::endl;
  (void)sport_client.RecoveryStand();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  (void)sport_client.StandUp();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  (void)sport_client.BalanceStand();
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  std::cout << "[demo] move in standing mode: vx=" << vx
            << " vy=" << vy << " vyaw=" << vyaw
            << " duration=" << duration_sec << "s" << std::endl;
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    (void)sport_client.Move(vx, vy, vyaw);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::steady_clock::now() - start);
    if (elapsed.count() >= duration_sec) {
      break;
    }
  }
  (void)sport_client.StopMove();
  std::cout << "[demo] done." << std::endl;
  return 0;
}
