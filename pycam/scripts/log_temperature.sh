#!/bin/bash
# Scipt to log temperature and save to path
# The script logs tmperature just once, so should be run every time temperature needs to be logged (use scheduler)
#
# This script borrow much of the code from wittypi/utilities.sh

export LC_ALL=en_GB.UTF-8

if [ -z ${I2C_RTC_ADDRESS+x} ]; then
	readonly I2C_RTC_ADDRESS=0x68
	readonly I2C_MC_ADDRESS=0x69

	readonly I2C_VOLTAGE_IN_I=1
	readonly I2C_VOLTAGE_IN_D=2
	readonly I2C_VOLTAGE_OUT_I=3
	readonly I2C_VOLTAGE_OUT_D=4
	readonly I2C_CURRENT_OUT_I=5
	readonly I2C_CURRENT_OUT_D=6
	readonly I2C_POWER_MODE=7
	readonly I2C_LV_SHUTDOWN=8

	readonly I2C_CONF_ADDRESS=9
	readonly I2C_CONF_DEFAULT_ON=10
	readonly I2C_CONF_PULSE_INTERVAL=11
	readonly I2C_CONF_LOW_VOLTAGE=12
	readonly I2C_CONF_BLINK_LED=13
	readonly I2C_CONF_POWER_CUT_DELAY=14
	readonly I2C_CONF_RECOVERY_VOLTAGE=15
	readonly I2C_CONF_DUMMY_LOAD=16
	readonly I2C_CONF_ADJ_VIN=17
	readonly I2C_CONF_ADJ_VOUT=18
	readonly I2C_CONF_ADJ_IOUT=19

	readonly HALT_PIN=4    # halt by GPIO-4 (BCM naming)
	readonly SYSUP_PIN=17  # output SYS_UP signal on GPIO-17 (BCM naming)
fi

wittypi_home="`dirname \"$0\"`"
wittypi_home="`( cd \"$wittypi_home\" && pwd )`"
log2file()
{
  local datetime=$(date +'[%Y-%m-%d %H:%M:%S]')
  local msg="$datetime $1"
  echo $msg >> $wittypi_home/wittyPi.log
}

log()
{
  if [ $# -gt 1 ] ; then
    echo $2 "$1"
  else
    echo "$1"
  fi
  log2file "$1"
}

i2c_read()
{
  local retry=0
  if [ $# -gt 3 ] ; then
    retry=$4
  fi
  local result=$(i2cget -y $1 $2 $3)
  if [[ $result =~ ^0x[0-9a-fA-F]{2}$ ]] ; then
    echo $result;
  else
    retry=$(( $retry + 1 ))
    if [ $retry -eq 4 ] ; then
      log "I2C read $1 $2 $3 failed (result=$result), and no more retry."
    else
      sleep 1
      log2file "I2C read $1 $2 $3 failed (result=$result), retrying $retry ..."
      i2c_read $1 $2 $3 $retry
    fi
  fi
}

i2c_write()
{
  local retry=0
  if [ $# -gt 4 ] ; then
    retry=$5
  fi
  i2cset -y $1 $2 $3 $4
  local result=$(i2c_read $1 $2 $3)
  if [ "$result" != $(dec2hex "$4") ] ; then
    retry=$(( $retry + 1 ))
    if [ $retry -eq 4 ] ; then
      log "I2C write $1 $2 $3 $4 failed (result=$result), and no more retry."
    else
      sleep 1
      log2file "I2C write $1 $2 $3 $4 failed (result=$result), retrying $retry ..."
      i2c_write $1 $2 $3 $4 $retry
    fi
  fi
}

dec2hex()
{
  printf "0x%02x" $1
}

get_temperature()
{
  local ctrl=$(i2c_read 0x01 $I2C_RTC_ADDRESS 0x0E)
  i2c_write 0x01 $I2C_RTC_ADDRESS 0x0E $(($ctrl|0x20))
  sleep 0.2
  local t1=$(i2c_read 0x01 $I2C_RTC_ADDRESS 0x11)
  local t2=$(i2c_read 0x01 $I2C_RTC_ADDRESS 0x12)
  local sign=$(($t1&0x80))
  local c=''
  if [ $sign -ne 0 ] ; then
    c+='-'
    c+=$((($t1^0xFF)+1))
  else
    c+=$(($t1&0x7F))
  fi
  c+='.'
  c+=$(((($t2&0xC0)>>6)*25))
  echo -n "$c$(echo $'\xc2\xb0'C)"
  if hash awk 2>/dev/null; then
    local f=$(awk "BEGIN { print $c*1.8+32 }")
    echo " / $f$(echo $'\xc2\xb0'F)"
  else
    echo ''
  fi
}

# Get temperature
log_file='/home/pi/pycam/logs/temperature.log'
temperature=$(date +"%F %T")
temperature+=" "

temp="$(get_temperature)"
temp_split=`echo "$temp" | cut -d "/" -f 1`

# Create final line
temperature+="$temp_split"

# Write temperature to log file
echo "$temperature" >> "$log_file"
echo "$temperature"
